//===- Validation.cpp - C/C++ Validation-----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/PassesEnums.cpp.inc"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_VALIDATION
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"
} // namespace emitc
} // namespace mlir

using namespace mlir;
using namespace mlir::emitc;

namespace {

static LogicalResult callOpHasNoTemplateArguments(CallOpaqueOp op) {
  if (op.getTemplateArgs())
    return failure();

  return success();
}

static LogicalResult funcOpHasMultipleReturnValues(func::FuncOp op) {
  if (op.getNumResults() > 1)
    return success();

  return failure();
}

static LogicalResult callOpaqueOpHasFloatTemplateArgs(CallOpaqueOp op) {
  if (op.getTemplateArgs()) {
     auto templateArgs = op.getTemplateArgs();
    for (auto templateArg : templateArgs.value()) {
      if (isa<FloatAttr>(templateArg))
        return success();
    }
  }

  return failure();
}


/*
template <typename iterator_range>
static bool hasTensorType(iterator_range types) {
  return llvm::any_of(types, [](Type t) { return isa<TensorType>(t); });
}
*/

static bool usesTensorTypes(Operation *op) {

  //Operation::operand_type_range operandTypes = op->getOperandTypes();

  /*
  auto operandsHasTensorType = hasTensorType(op->getOperandTypes());
  auto resultsHasTensorVectorType = hasTensorType(op->getResultTypes());
  if ((operandsHasTensorType) || (resultsHasTensorVectorType))
    return true;
  */

  return false;

  /*
  for(auto it = operandTypes.begin(); std::next(it) != operandTypes.end(); ++it ) {
    *it->getT
    return true;
  }



  //for( operandType = operandTypes.begin()  : operandTypes) {
  //  if (isa<TensorType>(operandType))
      return true;
  //}


  auto resultTypes = op->getResultTypes();


  return true;


  return false;

  */
}

//static bool funcDeclarationOrDefinitionBeforeCall() {
//}

struct Validation : public emitc::impl::ValidationBase<Validation> {
public:
  explicit Validation(){};
  explicit Validation(const ValidationOptions &options) : Validation() {
    this->standard = options.standard;
  }
  void runOnOperation() final;
};

void Validation::runOnOperation() {
  // configLevelAndProfile();
  getOperation().walk([&](Operation *op) {

    if (standard == LanguageStandard::C99) {
      if (auto callOp = dyn_cast<CallOpaqueOp>(op)) {
        if (failed(callOpHasNoTemplateArguments(callOp))) {
          return signalPassFailure();
        }
      } else {
        if (usesTensorTypes(op)) {
          return signalPassFailure();
        }
      }
    }

    if (standard == LanguageStandard::Cpp98) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (succeeded(funcOpHasMultipleReturnValues(funcOp))) {
          return signalPassFailure();
        }
      } else if (auto funcOp = dyn_cast<CallOpaqueOp>(op)) {
        if (succeeded(callOpaqueOpHasFloatTemplateArgs(funcOp))) {
          return signalPassFailure();
        }
      }
    }

    /*LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<emitc::AddOp, emitc::ApplyOp, emitc::AssignOp,
                  emitc::CallOpaqueOp, emitc::CastOp, emitc::CmpOp,
                  emitc::ConstantOp, emitc::DivOp, emitc::ExpressionOp,
                  emitc::ForOp, emitc::IfOp, emitc::IncludeOp, emitc::MulOp,
                  emitc::RemOp, emitc::SubOp, emitc::VariableOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<emitc::LiteralOp>([&](auto op) { return success(); })
            .Default([&](Operation *) {
              return op.emitOpError("unable to find printer for op");
            });
      */
  });
}
} // namespace
