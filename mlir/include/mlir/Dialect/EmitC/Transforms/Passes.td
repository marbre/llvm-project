//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
#define MLIR_DIALECT_EMITC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/IR/EnumAttr.td"

def FormExpressions : Pass<"form-expressions"> {
  let summary = "Form C-style expressions from C-operator ops";
  let description = [{
    The pass wraps emitc ops modelling C operators in emitc.expression ops and
    then folds single-use expressions into their users where possible.
  }];
  let constructor = "mlir::emitc::createFormExpressionsPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
}

def LanguageStandard : I32EnumAttr<"LanguageStandard", "C or C++ standard",
    [
      I32EnumAttrCase<"C99", 0, "C99">,
      I32EnumAttrCase<"Cpp98", 1, "C++98 / C++03">,
      I32EnumAttrCase<"Cpp11", 2, "C++11">,
      I32EnumAttrCase<"Cpp20", 3, "C++20">
    ]>{
  let cppNamespace = "mlir::emitc";
  let genSpecializedAttr = 0;
}

def Validation : Pass<"emitc-validation", "mlir::ModuleOp"> {
  let summary = "Validates C/C++ standards";
  let description = [{
    This pass valides if the EmitC input operations match the specification for
    a given C or C++ standard.
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
  //let constructor = "mlir::emitc::createValidation()";
  let options = [
    Option<"standard", "standard", "mlir::emitc::LanguageStandard",
            /*default=*/"mlir::emitc::LanguageStandard::Cpp20",
            "Select the C or C++ standard to validate for",
            [{::llvm::cl::values(
              clEnumValN(mlir::emitc::LanguageStandard::C99,
                         "c99",
                         "C99 standard."),
              clEnumValN(mlir::emitc::LanguageStandard::Cpp98,
                         "c++98",
                         "C++98 / C++03 standard."),
              clEnumValN(mlir::emitc::LanguageStandard::Cpp11,
                         "c++11",
                         "C++11 standard."),
              clEnumValN(mlir::emitc::LanguageStandard::Cpp20,
                         "c++20",
                         "C++20 standard.")
            )}]>
  ];
}

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
