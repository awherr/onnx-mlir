// SPDX-License-Identifier: Apache-2.0

//===--------------------- Nx.td - MLIR Operations ---------*- tablegen -===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains TableGen definition of nx operations.
//
// REQUEST:
// When you perform a change to the DEN dialect, please follow the steps below
// to update the Nx dialect documentation.
//
// 1) cd onnx-mlir/build
// 2) make onnx-mlir-docs
// 3) the newly build nx.md file resides in the docs/Dialects directory.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "src/Interface/SpecializedKernelOpInterface.td"

def Nx_Dialect : Dialect {
  let name = "nx";
  let cppNamespace = "::mlir";
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "shape::ShapeDialect",
  ];
}

def StringType : Type<CPred<"mlir::isa<nx::StringType>($_self)">, "string type">;

// Require regions to have nx.terminate terminator operation.
def ImplicitNxTerminator : SingleBlockImplicitTerminator<"NxTerminatorOp">;

def NxEndOp : Op<Nx_Dialect, "end", [Pure, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "";
  let description = [{}];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

// def NxJmpOp : Op<Nx_Dialect, "jmp", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxJmpCOp : Op<Nx_Dialect, "jmp_c", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxJmpNCOp : Op<Nx_Dialect, "jmp_nc", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxNextOp : Op<Nx_Dialect, "next", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxRepeatOp : Op<Nx_Dialect, "repeat", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxPrevOp : Op<Nx_Dialect, "prev", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxSkipOp: Op<Nx_Dialect, "skip", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxSkipCOp: Op<Nx_Dialect, "skip_c", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];

//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxSkipNCOp: Op<Nx_Dialect, "skip_nc", [Pure, Terminator, ReturnLike,
//     MemRefsNormalizable]> {
//   let summary = "";
//   let description = [{}];
//   let assemblyFormat = "attr-dict";
//   let hasVerifier = 1;
// }

// def NxRMWCAOp: Op<Nx_Dialect, "rmw_ca", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = (ins SignlessIntegerLike:$arg);
//   let assemblyFormat = [{
//     $arg attr-dict `:` type($arg)
//   }];
// }

// def NxRMWDAOp: Op<Nx_Dialect, "rmw_da", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = (ins SignlessIntegerLike:$arg);
//   let assemblyFormat = [{
//     $arg attr-dict `:` type($arg)
//   }];
// }

// def NxRDCDAOp: Op<Nx_Dialect, "rdc_da", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = (ins SignlessIntegerLike:$arg);
//   let assemblyFormat = [{
//     $arg attr-dict `:` type($arg)
//   }];
// }

// def NxSpikeOp: Op<Nx_Dialect, "spike", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = (ins SignlessIntegerLike:$arg);
//   let assemblyFormat = [{
//     $arg attr-dict `:` type($arg)
//   }];
// }

// def NxProbeOp: Op<Nx_Dialect, "probe", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = (ins SignlessIntegerLike:$arg);
//   let assemblyFormat = [{
//     $arg attr-dict `:` type($arg)
//   }];
// }


// def DEN_NONE   : I32BitEnumAttrCaseBit<"always",0>;
// def DEN_CMP_EQ : I32BitEnumAttrCaseBit<"eq", 1>;
// def DEN_CMP_NE : I32BitEnumAttrCaseBit<"ne", 2>;
// def DEN_CMP_LT : I32BitEnumAttrCaseBit<"lt", 3>;
// def DEN_CMP_GT : I32BitEnumAttrCaseBit<"gt", 4>;
// def DEN_CMP_LE : I32BitEnumAttrCaseBit<"le", 5>;
// def DEN_CMP_GE : I32BitEnumAttrCaseBit<"ge", 6>;
// def NxCmpFlags : I32BitEnumAttr<
//     "NxCmpModeFlags",
//     "Compare flags",
//     [DEN_CMP_EQ,DEN_CMP_NE,DEN_CMP_LT, DEN_CMP_GT, DEN_CMP_LE, DEN_CMP_GE]> {
//   let separator = ",";
//   let cppNamespace = "::mlir::nx";
//   let genSpecializedAttr = 0;
//   let printBitEnumPrimaryGroups = 1;
// }

// def Nx_CmpFlags :
//     EnumAttr<Nx_Dialect, NxCmpFlags, "cmp_flag"> {
//   let assemblyFormat = "`<` $value `>`";
// }

// def NxCmpJmp: Op<Nx_Dialect, "cmp_jmp", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = ( ins 
//       DefaultValuedAttr<Nx_CmpFlags, "::mlir::nx::NxCmpFlags::always">:$mode,
//       Arg<AnyMemRef, "the reference to load from", [MemRead]>:$cond
//       );
//   let assemblyFormat = "(`mode` `` $mode^)? $cond attr-dict `:` type($cond)";
// }

// def NxCmpSkp: Op<Nx_Dialect, "cmp_skp", [Pure]> {
//   let summary = "";
//   let description = [{}];
//   let hasVerifier = 1;
//   let arguments = ( ins 
//       DefaultValuedAttr<Nx_CmpFlags, "::mlir::nx::NxCmpFlags::always">:$mode,
//       Arg<AnyMemRef, "the reference to load from", [MemRead]>:$cond
//       );
//   let assemblyFormat = "(`mode` `` $mode^)? $cond attr-dict `:` type($cond)";
// }

// class Nx_Op<string mnemonic, list<Trait> traits = []> :
//     Op<Nx_Dialect, mnemonic,
//        traits #
//        [NoMemoryEffect] #
//        ElementwiseMappable.traits>;

// class Nx_ArithOp<string mnemonic, list<Trait> traits = []> :
//     Nx_Op<mnemonic, traits>;

// class Nx_RMWLastOp<string mnemonic, list<Trait> traits = []> :
//     Nx_ArithOp<mnemonic, traits # [Pure]> {
//   let arguments = (ins 
//     Arg<AnyMemRef, "the reference to load from", [MemRead]>:$src);
//   let assemblyFormat = "$src attr-dict `:` type($src)";
// }

// class Nx_RMWBinaryOp<string mnemonic, list<Trait> traits = []> :
//     Nx_Op<mnemonic, traits #
//         [MemRefsNormalizable]> {
//   let arguments = (ins 
//     Arg<AnyMemRef, "the reference to store to",  [MemWrite]>:$dst,
//     Arg<AnyMemRef, "the reference to load from", [MemRead]>:$src);
//   let assemblyFormat = "$dst `,` $src attr-dict `:` `(` type($dst) `,` type($src) `)`";
// }

// def Nx_AddOp : Nx_RMWBinaryOp<"add"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_AddSatOp : Nx_RMWBinaryOp<"add_sat"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Adl : Nx_RMWLastOp<"adl"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_AdlSat : Nx_RMWLastOp<"adl_sat"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_AddCOp : Nx_RMWBinaryOp<"adc"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Sub : Nx_RMWBinaryOp<"sub"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Sbb : Nx_RMWBinaryOp<"sbb"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Neg : Nx_RMWBinaryOp<"neg"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Abs : Nx_RMWBinaryOp<"abs"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_SubGE : Nx_RMWBinaryOp<"sub_ge"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Min : Nx_RMWBinaryOp<"min"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Max : Nx_RMWBinaryOp<"max"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Mov : Nx_RMWBinaryOp<"mov"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Sel : Nx_RMWBinaryOp<"sel"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Bit : Nx_RMWBinaryOp<"bit"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_And : Nx_RMWBinaryOp<"and"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Nand : Nx_RMWBinaryOp<"nand"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Or : Nx_RMWBinaryOp<"or"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Nor : Nx_RMWBinaryOp<"nor"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Xor : Nx_RMWBinaryOp<"xor"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Xnor : Nx_RMWBinaryOp<"xnor"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Not : Nx_RMWBinaryOp<"not"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Clz : Nx_RMWBinaryOp<"clz"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// def Nx_Mul : Nx_RMWBinaryOp<"mul"> {
//   let summary = "";
//   let description = [{ }];
//   let hasFolder = 1;
// }

// Bit 
// def NxCallOp : Op<Nx_Dialect, "call",
//     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
//   > {
//   let summary = "call operation";
//   let description = [{
//     The call operation provides a generic way to replace an ONNX Op with a call
//     to an external function at Nx level.
//     `funcName` attributes determines which function to call.
//     `parameters` is the inputs to Nx.Call. It includes the outputs and inputs
//     of the ONNX Op. The outputs and inputs are already lowered to MemRefs.
//     The external function is assumed NOT to allocate or free any memory.
//     'numOfOutput` attribute to tell how manu outputs Memref in parameters.
//     mlir::OpTrait::AttrSizedOperandSegments is not used to put outputs and
//     inputs into separate variadic parameters because I am thinking of mixing
//     the inputs and outpus as required by external library.

//     The attributes of the ONNX Op will be copied to NxCallOp under the control
//     of the user.
//     In Nx To llvm lowering, the parameters and attributes will be lowered to
//     parameters of the llvm function call.

//     Several builder is defined to help translating an ONNX Op to Nx.Call.
//     User can provides the allocated MemRefs for outputs and the inputs
//     separately. The inputs are usually the operands of the ONNX Op.
//     The attributes of ONNX Op can be copied or not copied based on a bool
//     parameter in the builder. Builder also provide a mechanism for user
//     to selectively copy some attributes.

//     The nx.call op will be lowered to llvm at nx-to-llvm conversion in which
//     OMTensor is used as a container for MemRef arguments. Other representation
//     of parameters, such as data pointer only, will be supported in future.
//   }];

//   let arguments = (ins StrAttr:$funcName,
//       // Should mlir::OpTrait::AttrSizedOperandSegments be used?
//       // The case for nx.Call is simple.
//       DefaultValuedAttr<SI64Attr, "1">:$numOfOutput,
//       Variadic<AnyType>:$parameters);

//   // Return Value for the Call.
//   // No return if the type is NoneType (void in llvm)
//   // Only scalar type is supported now.
//   // In future, return of memref can be supported with pointer of OMTensor.
//   // The returned memref will be created inside the call. 
//   let results = (outs Variadic<AnyTypeOf<[AnyFloat, AnyInteger]>>:$returnValue);

//   // builders to build NxCallOp from op and operands, helping conversion from
//   // onnx to nx.
//   // The name of function can be determined by the op name and elemnt type of
//   // the return, or given to builder if the simple rule does not work.
//   // Attributes of the op will be propagated to NxCallOp if the copyAttrs is
//   // true. Or the attribute names can be specified.
//   let builders = [
//       OpBuilder<(ins "std::string":$funcNameStr, "int64_t":$numOfOutput, "mlir::ValueRange":$operands)>,
//       OpBuilder<(ins "mlir::StringAttr":$funcNameStr, "IntegerAttr":$numOfOutput, "mlir::ValueRange":$operands)>,
//       OpBuilder<(ins "std::string":$funcNameStr, "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "std::vector<std::string>":$attributeNames)>,
//     OpBuilder<(ins "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "bool":$copyAttrs)>,
//       OpBuilder<(ins "std::string":$funcNameStr, "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "std::vector<std::string>":$attributeNames)>,
//       OpBuilder<(ins "std::string":$funcNameStr, "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "bool":$copyAttrs)>];
// }
