// SPDX-License-Identifier: Apache-2.0

//===--------------------- NxKernel.td - MLIR Operations ---------*- tablegen -===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains TableGen definition of nx operations.
//
// REQUEST:
// When you perform a change to the DEN dialect, please follow the steps below
// to update the NxKernel dialect documentation.
//
// 1) cd onnx-mlir/build
// 2) make onnx-mlir-docs
// 3) the newly build nx.md file resides in the docs/Dialects directory.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "src/IR/AttrBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
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


def NxKernel_Dialect : Dialect {
  let name = "nxk";
  let cppNamespace = "::mlir";
  let usePropertiesForAttributes = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
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
class NxKernel_Attr<string name, list<Trait> traits = []>
	: BaseLayoutAttr<NxKernel_Dialect, name, traits>;

def NXKSpikeTypeEnum : I32EnumAttr<"SpikeType", "", [
                                I32EnumAttrCase<"eq", 0>,
                                I32EnumAttrCase<"lu", 1>,
                                I32EnumAttrCase<"ws", 2>,
                                I32EnumAttrCase<"wu", 3>,
                                I32EnumAttrCase<"bs", 4>,
                                I32EnumAttrCase<"bu", 5>,
                                I32EnumAttrCase<"zs", 6>,
                                I32EnumAttrCase<"zu", 7>,
                                I32EnumAttrCase<"z",  8>]> {
  let genSpecializedAttr = 0;
}
def NXKSpikeTypeAttr : EnumAttr<NxKernel_Dialect, NXKSpikeTypeEnum, "">;

def NXKSignModeEnum : I32EnumAttr<"SignMode", "", [
                     I32EnumAttrCase<"null",  0>,
                     I32EnumAttrCase<"mixed", 1>,
                     I32EnumAttrCase<"inhibitory", 2>,
                     I32EnumAttrCase<"excitatory", 3>]>{
  let genSpecializedAttr = 0;
}

def NXKSignModeAttr : EnumAttr<NxKernel_Dialect, NXKSignModeEnum, "">;

def KernelType : Type<CPred<"mlir::isa<nx::KernelType>($_self)">, "kernel type">;
def ModuleType : Type<CPred<"mlir::isa<nx::ModuleType>($_self)">, "module type">;
def SynapseType : Type<CPred<"mlir::isa<nx::SynapseType>($_self)">, "synapse type">;

def NxTensorEncodingAttr : NxKernel_Attr<"NxTensorEncoding"> {
  let hasCustomAssemblyFormat = 1;

  let description = [{
  }];

  // Data in tensor encoding.
  let parameters = (ins
    // A original data layout of the pre-stickified data.
    "NxTensorEncodingAttr::DataLayout":$dataLayout,
    "NxTensorEncodingAttr::QuantizedType":$quantizedType
  );

  let extraClassDeclaration = [{
    enum class DataLayout {
      UNDEFINED, _1D, _2D, _2DS, _3D, _3DS, _4D, _4DS,
      NCHW, NHWC, HWCK,
      FICO, ZRH, BFICO, BZRH
    };
    enum class QuantizedType {
      UNDEFINED, DLFLOAT16, INT8, WEIGHTS
    };
    // QuantizedType is optional.
    static NxTensorEncodingAttr get(::mlir::MLIRContext *context, NxTensorEncodingAttr::DataLayout dataLayout) {
      return get(context, dataLayout, NxTensorEncodingAttr::QuantizedType::UNDEFINED);
    }
  }];
}

class DataLayoutOfPred<string layout> : And<[
  CPred<"(mlir::cast<::mlir::RankedTensorType>($_self)) &&"
        "(mlir::dyn_cast_or_null<NxTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding())) &&"
        "(mlir::cast<NxTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding()).getDataLayout()"
        " == NxTensorEncodingAttr::DataLayout::" # layout # ")">
]>;

// Whether a ztensor type has the specified quantized type.
class QuantizedTypeOfPred<string qtype> : And<[
  CPred<"(mlir::cast<::mlir::RankedTensorType>($_self)) &&"
        "(mlir::dyn_cast_or_null<NxTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding())) &&"
        "(mlir::cast<NxTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding()).getQuantizedType()"
        " == NxTensorEncodingAttr::QuantizedType::" # qtype # ")">
]>;

def UnrankedNxTensor : UnrankedTensorOf<[I32]>;

def UnrankedQNxTensor : UnrankedTensorOf<[I8]>;

// Whether a shaped type has one of the specified quantized type.
class HasAnyQuantizedTypeOfPred<list<string> qtypes> : And<[
  Or<!foreach(qtype, qtypes, QuantizedTypeOfPred<qtype>)>]>;


class NxTensorOf<string layout, list<int> ranks> :
    Type<And<[TensorOf<[F16]>.predicate, HasAnyRankOfPred<ranks>,
              DataLayoutOfPred<layout>]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         TensorOf<[F16]>.summary # " with layout " # layout,
         "::mlir::TensorType">;

// Quantized NxTensor.
class QNxTensorOf<string layout, list<int> ranks> :
    Type<And<[TensorOf<[I8, F16]>.predicate,
              DataLayoutOfPred<layout>,
              HasAnyQuantizedTypeOfPred<["DLFLOAT16", "INT8", "WEIGHTS"]>,
              HasAnyRankOfPred<ranks>
             ]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         TensorOf<[I8, F16]>.summary # " with layout " # layout,
         "::mlir::TensorType">;

def NxTensor_1D: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_1D", [1]>]>;
def NxTensor_2D: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_2D", [2]>]>;
def NxTensor_2DS: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_2DS", [2]>]>;
def NxTensor_3D: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_3D", [3]>]>;
def NxTensor_3DS: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_3DS", [3]>]>;
def NxTensor_4D: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_4D", [4]>]>;
def NxTensor_4DS: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"_4DS", [4]>]>;
def NxTensor_NHWC: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"NHWC", [4]>]>;
def NxTensor_NCHW: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"NCHW", [4]>]>;
def NxTensor_HWCK: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"HWCK", [4]>]>;
def NxTensor_FICO: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"FICO", [2, 3]>]>;
def NxTensor_ZRH: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"ZRH", [2, 3]>]>;
def NxTensor_BFICO: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"BFICO", [2, 3]>]>;
def NxTensor_BZRH: AnyTypeOf<[UnrankedNxTensor, NxTensorOf<"BZRH", [2, 3]>]>;

def StringType : Type<CPred<"mlir::isa<nx::StringType>($_self)">, "string type">;

// Require regions to have nx.terminate terminator operation.
def ImplicitNxKernelTerminator : SingleBlockImplicitTerminator<"NxKernelTerminatorOp">;

def NXKSynapseConv : Op<NxKernel_Dialect, "synapse_conv", []> {
  let summary = "";
  let description = [{}];

  let arguments = (ins I64ArrayAttr:$kernel_shape,
                       NxTensor_HWCK:$weights,
                       NxTensor_HWCK:$delay,
                       NxTensor_HWCK:$tag,
                       DefaultValuedOptionalAttr<I32ArrayAttr,"{1,1}">:$strides,
                       DefaultValuedOptionalAttr<I32ArrayAttr,"{0,0}">:$padding, 
                       DefaultValuedOptionalAttr<I32ArrayAttr,"{1,1}">:$dialation,
                       DefaultValuedAttr<I32Attr, "1">:$group,
                       OptionalAttr<I32Attr>:$num_weight_bits,
                       DefaultValuedAttr<I32Attr, "0">:$num_delay_bits,
                       DefaultValuedAttr<I32Attr, "0">:$num_tag_bits,
                       DefaultValuedAttr<I32Attr, "0">:$weight_exp,
                       OptionalAttr<NXKSignModeAttr>:$sign_mode,
                       OptionalAttr<I32Attr>:$sparse_packing,
                       OptionalAttr<I32Attr>:$optimize_weights,
                       DefaultValuedStrAttr<StrAttr, "">:$name
                       );
  let results = (outs KernelType:$output);
  let hasVerifier = 0;
}

def NXKSynapseMaxOp : Op<NxKernel_Dialect, "synapse_max", []> {
  let summary = "";
  let description = [{}];
  let arguments = (ins I64ArrayAttr:$shape,
                       I32ArrayAttr:$num_args,
                       NXKSpikeTypeAttr:$out_spike_type
                       );
  let results = (outs KernelType:$output);

  let hasVerifier = 1;
}

def NXKSynapseLinear : Op<NxKernel_Dialect, "synapse_linear", []> {
  let summary = "";
  let description = [{}];
  let arguments = (ins NxTensor_HWCK:$weights,
                       NxTensor_HWCK:$delay,
                       NxTensor_HWCK:$tag,
                       OptionalAttr<I32Attr>:$num_weight_bits,
                       DefaultValuedAttr<I32Attr, "0">:$num_delay_bits,
                       DefaultValuedAttr<I32Attr, "0">:$num_tag_bits,
                       DefaultValuedAttr<I32Attr, "0">:$weight_exp,
                       OptionalAttr<NXKSignModeAttr>:$sign_mode,
                       OptionalAttr<I32Attr>:$sparse_packing,
                       OptionalAttr<I32Attr>:$optimize_weights,
                       OptionalAttr<I32Attr>:$use_shared_axon,
                       DefaultValuedStrAttr<StrAttr, "">:$name
                       );
  let results = (outs KernelType:$output);
  let hasVerifier = 0;
}

def NXKSynapsePool : Op<NxKernel_Dialect, "synapse_pool", []> {
  let summary = "";
  let description = [{}];
  let arguments = (ins I64ArrayAttr:$output_shape,
                       I64ArrayAttr:$kernel_size,
                       DefaultValuedOptionalAttr<I32ArrayAttr,"{1,1}">:$strides, 
                       DefaultValuedOptionalAttr<I32ArrayAttr,"{0,0}">:$padding, 
                       DefaultValuedOptionalAttr<I32ArrayAttr,"{1,1}">:$dialation,
                       OptionalAttr<I32Attr>:$sparse_packing,
                       DefaultValuedStrAttr<StrAttr, "">:$name
                       );
  let results = (outs KernelType:$output);
  let hasVerifier = 0;
}

