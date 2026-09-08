// Host-only observer of the native A5 pipeline through PlanMemory.
// All configuration inference, pipeline construction and passes come from NPUIR.
#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/regbase/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"
#include "bishengir/Version/Version.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassInstrumentation.h"
#include <cstdlib>
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

namespace {
// End this observation process only after the unmodified local planner succeeds.
// Keeping the native pass instances also preserves non-CLI callbacks/options.
struct StopAfterLocalPlanMemory : mlir::PassInstrumentation {
  std::string outputPath;
  explicit StopAfterLocalPlanMemory(std::string path) : outputPath(std::move(path)) {}
  void runAfterPass(mlir::Pass *pass, mlir::Operation *operation) override {
    if (pass->getArgument() != "hivm-plan-memory-regbase") return;
    std::string description;
    llvm::raw_string_ostream stream(description);
    pass->printAsTextualPipeline(stream);
    if (description.find("mem-plan-mode=local-mem-plan") == std::string::npos) return;
    auto output = mlir::openOutputFile(outputPath);
    if (!output) std::exit(2);
    operation->print(output->os());
    output->os() << '\n';
    output->os().flush();
    output->keep();
    llvm::outs() << "PLAN_MEMORY_ORACLE_OK\n";
    llvm::outs().flush();
    llvm::errs().flush();
    std::exit(0);
  }
};
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  bishengir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerAllPasses();
  bishengir::registerAllPasses();
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  bishengir::BiShengIRCompileMainConfig::registerCLOptions();
  bishengir::registerPassManagerCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &out) {
    out << bishengir::getBiShengIRToolFullVersion("plan-memory-oracle") << '\n';
  });
  llvm::cl::ParseCommandLineOptions(argc, argv, "Native A5 PlanMemory observer\n");
  auto config = bishengir::BiShengIRCompileMainConfig::createFromCLOptions(true);
  if (config.getPureSimt() || config.getEnableSimdSimtMixCompile() ||
      config.shouldEnableCPURunner()) {
    llvm::errs() << "Oracle supports the SIMD A5 pipeline without CPU lowering.\n";
    return 2;
  }
  if (failed(checkInOutOptionsValidity(config)))
    return 2;
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(config.getInputFile(), &context);
  if (!module)
    return 2;
  mlir::ModuleOp input = *module;
  // These are the same pre-pipeline inference calls made by runRegBaseCompile.
  if (failed(bishengir::regbase::inferLayoutOptimization(input, config)) ||
      (config.getEnableTritonKernelCompile() &&
       failed(bishengir::regbase::inferMixedCV(input, config))))
    return 2;
  std::vector<std::string> arguments(argv + 1, argv + argc);
  config.setClArgs(bishengir::regbase::filterRegBaseForwardedHIVMCOptions(arguments));

  auto pipelineFile = mlir::openOutputFile(config.getOutputFile() + ".pipeline.txt");
  if (!pipelineFile) return 2;
  pipelineFile->keep();
  auto buildHIR = [&](mlir::PassManager &hir) {
    bishengir::regbase::buildBiShengHIRPipeline(hir, config);
    hir.printAsTextualPipeline(pipelineFile->os());
    pipelineFile->os() << '\n';
    pipelineFile->os().flush();
  };
  // The production runner installs FilterPassesAttr handling for backup
  // functions as well as verifier and pass-manager options.
  if (failed(bishengir::runPipeline(*module, buildHIR, config, "BiShengHIR")))
    return 1;

  auto buildFinal = [&](mlir::PassManager &native) {
  bishengir::regbase::buildFinalHIVMPipelines(native, config);
  std::string prefix;
  llvm::raw_string_ostream stream(prefix);
  stream << "builtin.module(";
  bool found = false, first = true;
  for (auto &pass : native.getPasses()) {
    if (!first) stream << ',';
    first = false;
    std::string serialized;
    llvm::raw_string_ostream passStream(serialized);
    pass.printAsTextualPipeline(passStream);
    stream << serialized;
    if (pass.getArgument() == "hivm-plan-memory-regbase" &&
        serialized.find("mem-plan-mode=local-mem-plan") != std::string::npos) {
      found = true;
      break;
    }
  }
  stream << ')';
  if (!found) {
    llvm::errs() << "Native final HIVM pipeline has no top-level local PlanMemory boundary.\n";
    std::exit(2);
  }
  pipelineFile->os() << prefix << '\n';
  pipelineFile->keep();

  pipelineFile->os().flush();
  native.addInstrumentation(std::make_unique<StopAfterLocalPlanMemory>(config.getOutputFile()));
  };
  if (failed(bishengir::runPipeline(*module, buildFinal, config, "buildFinalHIVMPipelines")))
    return 1;
  llvm::errs() << "Local PlanMemory completion was not observed.\n";
  return 2;
}
