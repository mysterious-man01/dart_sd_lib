import 'dart:ffi';
import 'dart:io';

import 'package:dart_sd_lib/bind/stable_diffusion.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';

class Sd {
  static stable_diffusion? _lib;

  // Pointer to sd instance
  late Pointer<sd_ctx_t> sdCtx;

  /// Getter for the Llama library.
  /// Loads the library based on the current platform
  stable_diffusion get lib{
    if(Platform.isAndroid || Platform.isLinux){
      _lib = stable_diffusion(DynamicLibrary.open('sdlib.so'));
    }
    /*else if(libPath != null && File(libPath!).existsSync()){
      _lib = stable_diffusion(DynamicLibrary.open(libPath!));
    }
    else{
        _lib = stable_diffusion(DynamicLibrary.process());
    }*/

    return _lib!;
  }

  /// Stable-Diffusion constructor
  /// 
  /// Initialize a instance of Stable-Diffusion
  Sd({
    required String modelPath,
    String vaePath = '',
    String taesdPath = '',
    String controlNetPath = '',
    String loraModelDir = '',       // Future attention
    String embedDir = '',           // Future attention
    String stackedIdEmbedDir = '',  // Future attention
    bool vaeDecodeOnly = false,
    bool vaeTiling = false,
    bool freeParamsImmediately = false,
    int nThreads = 1,
    int wtype = 34,
    int rngType = 0,
    int schedule = 2,
    bool keepClipOnCpu = false,
    bool keepControlNetCpu = false,
    bool keepVaeOnCpu = false,
    }){
    // Convert String to Pointer<Char>
    Pointer<Char> modelPathChar = modelPath.toNativeUtf8().cast<Char>();
    Pointer<Char> vaePathChar = vaePath.toNativeUtf8().cast<Char>();
    Pointer<Char> taesdPathChar = taesdPath.toNativeUtf8().cast<Char>();
    Pointer<Char> controlNetPathChar = controlNetPath.toNativeUtf8().cast<Char>();
    Pointer<Char> loraModelDirChar = loraModelDir.toNativeUtf8().cast<Char>();
    Pointer<Char> embedDirChar = embedDir.toNativeUtf8().cast<Char>();
    Pointer<Char> stackedIdEmbedDirChar = stackedIdEmbedDir.toNativeUtf8().cast<Char>();
    
    sdCtx = lib.new_sd_ctx(
      modelPathChar,
      vaePathChar,
      taesdPathChar,
      controlNetPathChar,
      loraModelDirChar,
      embedDirChar,
      stackedIdEmbedDirChar,
      vaeDecodeOnly,
      vaeTiling,
      freeParamsImmediately,
      nThreads,
      wtype,
      rngType,
      schedule,
      keepClipOnCpu,
      keepControlNetCpu,
      keepVaeOnCpu
    );

    // Clean up unused pointer variables
    malloc.free(modelPathChar);
    malloc.free(vaePathChar);
    malloc.free(taesdPathChar);
    malloc.free(controlNetPathChar);
    malloc.free(loraModelDirChar);
    malloc.free(embedDirChar);
    malloc.free(stackedIdEmbedDirChar);
  }

  /// Destructor
  /// 
  /// Clean up Stable-Diffusion instance
  void dispose(){
    lib.free_sd_ctx(sdCtx);
  }

  /// Generate image
  MemoryImage generate(String prompt,
    {String negativePrompt = '',
      dynamic fromImg,
      double strenght = 0.8,
      int clipSkip = 1,
      double cfgScale = 7,
      int width = 512,
      int height = 512,
      int sampleMethod = 0,
      int sampleSteps = 20,
      int seed = -1,
      int batchCount = 1,
      dynamic controlCond,
      double controlStrength = 0.0,
      double styleStrength = 0.0,
      bool normalizeInput = true,
      String inputIdImagesPath = '',
    }){
    if(controlCond != null){
      controlCond = nullptr;
    }

    Pointer<sd_image_t> result = nullptr;
    MemoryImage img;
    
    if(fromImg != null){
      result = lib.img2img(
        sdCtx,
        fromImg,
        prompt.toNativeUtf8().cast<Char>(),
        negativePrompt.toNativeUtf8().cast<Char>(),
        clipSkip,
        cfgScale,
        width,
        height,
        sampleMethod,
        sampleSteps,
        strenght,
        seed,
        batchCount,
        controlCond,
        controlStrength,
        styleStrength,
        normalizeInput,
        inputIdImagesPath.toNativeUtf8().cast<Char>()
      );
    }
    else{
      result = lib.txt2img(
        sdCtx,
        prompt.toNativeUtf8().cast<Char>(),
        negativePrompt.toNativeUtf8().cast<Char>(),
        clipSkip,
        cfgScale,
        width,
        height,
        sampleMethod,
        sampleSteps,
        seed,
        batchCount,
        controlCond,
        controlStrength,
        styleStrength,
        normalizeInput,
        inputIdImagesPath.toNativeUtf8().cast<Char>()
      );
    }

    int size = sizeOf<UnsignedChar>() * result.ref.channel * result.ref.width * result.ref.height;
    
    img = MemoryImage(result.ref.data.asTypedList(size));
    
    malloc.free(result);

    return img;
  }
}