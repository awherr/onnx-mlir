/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NxAccelerator.cpp -----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for the IBM Telum processor.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NX/Compiler/NxCompilerUtils.hpp"
#include "src/Accelerators/NX/Dialect/NX/Nx.hpp"
#include "src/Accelerators/NX/NxAccelerator.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include <memory>

#define DEBUG_TYPE "NxAccelerator"

extern llvm::cl::OptionCategory OMNxPassOptions;

namespace onnx_mlir {
namespace accel {

NxAccelerator *NxAccelerator::instance = nullptr;

NxAccelerator *NxAccelerator::getInstance() {
  if (instance == nullptr)
    instance = new NxAccelerator();
  return instance;
}

Accelerator *createNX() { return NxAccelerator::getInstance(); }
  
NxAccelerator::NxAccelerator() : Accelerator(Accelerator::Kind::NX) {
  LLVM_DEBUG(llvm::dbgs() << "Creating an Nx accelerator\n");
};

NxAccelerator::~NxAccelerator() { delete instance; }

// Return accelerator version number based on compile Nx version
uint64_t NxAccelerator::getVersionNumber() const {
  return 1;
}

void NxAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) const {
  LLVM_DEBUG(llvm::dbgs() << "Adding passes for Nx accelerator\n");
}

void NxAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering dialects for Nx accelerator\n");
}

void NxAccelerator::registerPasses(int optLevel) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering passes for Nx accelerator\n");
}

void NxAccelerator::configurePasses() const {
  LLVM_DEBUG(llvm::dbgs() << "Configuring passes for Nx accelerator\n");
  // configurePassesNx();
}

} // namespace accel
} // namespace onnx_mlir
