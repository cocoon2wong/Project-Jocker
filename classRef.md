<!--
 * @Author: Conghao Wong
 * @Date: 2022-06-23 09:30:53
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2022-06-23 09:30:53
 * @Description: file content
 * @Github: https://github.com/cocoon2wong
 * Copyright 2022 Conghao Wong, All Rights Reserved.
-->

# Classes Used in This Project

Packages:

<!-- GRAPH BEGINS HERE -->
```mermaid
    graph LR
        builtins_object("object(builtins)") --> codes.training.__vis_Visualization("Visualization(codes.training.__vis)")
        builtins_object("object(builtins)") --> codes.dataset.__dataset_VideoClip("VideoClip(codes.dataset.__dataset)")
        builtins_object("object(builtins)") --> codes.dataset.__agent_Agent("Agent(codes.dataset.__agent)")
        codes.__base_BaseObject("BaseObject(codes.__base)") --> codes.training.__structure_Structure("Structure(codes.training.__structure)")
        builtins_object("object(builtins)") --> builtins_type("type(builtins)")
        keras.engine.training_Model("Model(keras.engine.training)") --> codes.basemodels.__model_Model("Model(codes.basemodels.__model)")
        codes.__base_BaseObject("BaseObject(codes.__base)") --> codes.dataset.__manager_DatasetManager("DatasetManager(codes.dataset.__manager)")
        builtins_object("object(builtins)") --> codes.dataset.__dataset_Dataset("Dataset(codes.dataset.__dataset)")
        builtins_object("object(builtins)") --> codes.__base_BaseObject("BaseObject(codes.__base)")
        builtins_object("object(builtins)") --> codes.args.__args_BaseArgTable("BaseArgTable(codes.args.__args)")
        builtins_object("object(builtins)") --> codes.dataset.__trajectory_Trajectory("Trajectory(codes.dataset.__trajectory)")
        codes.__base_BaseObject("BaseObject(codes.__base)") --> codes.dataset.__maps_MapManager("MapManager(codes.dataset.__maps)")
        codes.__base_BaseObject("BaseObject(codes.__base)") --> codes.dataset.__manager_VideoClipManager("VideoClipManager(codes.dataset.__manager)")
        builtins_FileNotFoundError("FileNotFoundError(builtins)") --> codes.dataset.__manager_TrajMapNotFoundError("TrajMapNotFoundError(codes.dataset.__manager)")
        tqdm.utils_Comparable("Comparable(tqdm.utils)") --> tqdm.std_tqdm("tqdm(tqdm.std)")
        builtins_object("object(builtins)") --> wavetf._wavetf_WaveTFFactory("WaveTFFactory(wavetf._wavetf)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.transformer._utils_MultiHeadAttention("MultiHeadAttention(codes.basemodels.transformer._utils)")
        keras.engine.training_Model("Model(keras.engine.training)") --> codes.basemodels.transformer._transformer_TransformerEncoder("TransformerEncoder(codes.basemodels.transformer._transformer)")
        keras.engine.training_Model("Model(keras.engine.training)") --> codes.basemodels.transformer._transformer_Transformer("Transformer(codes.basemodels.transformer._transformer)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.transformer._transformer_EncoderLayer("EncoderLayer(codes.basemodels.transformer._transformer)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.transformer._transformer_Encoder("Encoder(codes.basemodels.transformer._transformer)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.transformer._transformer_DecoderLayer("DecoderLayer(codes.basemodels.transformer._transformer)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.transformer._transformer_Decoder("Decoder(codes.basemodels.transformer._transformer)")
        codes.basemodels.layers.__transformLayers__BaseTransformLayer("_BaseTransformLayer(codes.basemodels.layers.__transformLayers)") --> codes.basemodels.layers.__transformLayers_InverseHaar1D("InverseHaar1D(codes.basemodels.layers.__transformLayers)")
        codes.basemodels.layers.__transformLayers__BaseTransformLayer("_BaseTransformLayer(codes.basemodels.layers.__transformLayers)") --> codes.basemodels.layers.__transformLayers_InverseDB2_1D("InverseDB2_1D(codes.basemodels.layers.__transformLayers)")
        codes.basemodels.layers.__transformLayers__BaseTransformLayer("_BaseTransformLayer(codes.basemodels.layers.__transformLayers)") --> codes.basemodels.layers.__transformLayers_IFFTLayer("IFFTLayer(codes.basemodels.layers.__transformLayers)")
        codes.basemodels.layers.__transformLayers__BaseTransformLayer("_BaseTransformLayer(codes.basemodels.layers.__transformLayers)") --> codes.basemodels.layers.__transformLayers_Haar1D("Haar1D(codes.basemodels.layers.__transformLayers)")
        codes.basemodels.layers.__transformLayers__BaseTransformLayer("_BaseTransformLayer(codes.basemodels.layers.__transformLayers)") --> codes.basemodels.layers.__transformLayers_FFTLayer("FFTLayer(codes.basemodels.layers.__transformLayers)")
        codes.basemodels.layers.__transformLayers__BaseTransformLayer("_BaseTransformLayer(codes.basemodels.layers.__transformLayers)") --> codes.basemodels.layers.__transformLayers_DB2_1D("DB2_1D(codes.basemodels.layers.__transformLayers)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.layers.__traj_TrajEncoding("TrajEncoding(codes.basemodels.layers.__traj)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.layers.__traj_ContextEncoding("ContextEncoding(codes.basemodels.layers.__traj)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.layers.__linear_LinearLayer("LinearLayer(codes.basemodels.layers.__linear)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.layers.__linear_LinearInterpolation("LinearInterpolation(codes.basemodels.layers.__linear)")
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> codes.basemodels.layers.__graphConv_GraphConv("GraphConv(codes.basemodels.layers.__graphConv)")
```
```mermaid
    graph LR
        keras.engine.base_layer_Layer("Layer(keras.engine.base_layer)") --> silverballers.__layers_OuterLayer("OuterLayer(silverballers.__layers)")
        builtins_object("object(builtins)") --> builtins_type("type(builtins)")
        codes.args.__args_BaseArgTable("BaseArgTable(codes.args.__args)") --> silverballers.__args_HandlerArgs("HandlerArgs(silverballers.__args)")
        silverballers.handlers.__baseHandler_BaseHandlerModel("BaseHandlerModel(silverballers.handlers.__baseHandler)") --> silverballers.handlers.__burnwoodC_BurnwoodCModel("BurnwoodCModel(silverballers.handlers.__burnwoodC)")
        silverballers.handlers.__baseHandler_BaseHandlerStructure("BaseHandlerStructure(silverballers.handlers.__baseHandler)") --> silverballers.handlers.__burnwoodC_BurnwoodC("BurnwoodC(silverballers.handlers.__burnwoodC)")
        codes.training.__structure_Structure("Structure(codes.training.__structure)") --> silverballers.handlers.__baseHandler_BaseHandlerStructure("BaseHandlerStructure(silverballers.handlers.__baseHandler)")
        codes.basemodels.__model_Model("Model(codes.basemodels.__model)") --> silverballers.handlers.__baseHandler_BaseHandlerModel("BaseHandlerModel(silverballers.handlers.__baseHandler)")
        tqdm.utils_Comparable("Comparable(tqdm.utils)") --> tqdm.std_tqdm("tqdm(tqdm.std)")
        codes.training.__structure_Structure("Structure(codes.training.__structure)") --> silverballers.agents.__baseAgent_BaseAgentStructure("BaseAgentStructure(silverballers.agents.__baseAgent)")
        codes.args.__args_BaseArgTable("BaseArgTable(codes.args.__args)") --> silverballers.__args_AgentArgs("AgentArgs(silverballers.__args)")
        keras.engine.training_Model("Model(keras.engine.training)") --> codes.basemodels.__model_Model("Model(codes.basemodels.__model)")
        codes.basemodels.__model_Model("Model(codes.basemodels.__model)") --> silverballers.agents.__agent47CE_Agent47CEModel("Agent47CEModel(silverballers.agents.__agent47CE)")
        silverballers.agents.__baseAgent_BaseAgentStructure("BaseAgentStructure(silverballers.agents.__baseAgent)") --> silverballers.agents.__agent47CE_Agent47CE("Agent47CE(silverballers.agents.__agent47CE)")
        codes.basemodels.__model_Model("Model(codes.basemodels.__model)") --> silverballers.agents.__agent47C_Agent47CModel("Agent47CModel(silverballers.agents.__agent47C)")
        silverballers.agents.__baseAgent_BaseAgentStructure("BaseAgentStructure(silverballers.agents.__baseAgent)") --> silverballers.agents.__agent47C_Agent47C("Agent47C(silverballers.agents.__agent47C)")
        silverballers.__baseSilverballers_BaseSilverballers("BaseSilverballers(silverballers.__baseSilverballers)") --> silverballers.__silverballers_Silverballers47C("Silverballers47C(silverballers.__silverballers)")
        codes.basemodels.__model_Model("Model(codes.basemodels.__model)") --> silverballers.__baseSilverballers_BaseSilverballersModel("BaseSilverballersModel(silverballers.__baseSilverballers)")
        codes.training.__structure_Structure("Structure(codes.training.__structure)") --> silverballers.__baseSilverballers_BaseSilverballers("BaseSilverballers(silverballers.__baseSilverballers)")
        codes.args.__args_BaseArgTable("BaseArgTable(codes.args.__args)") --> silverballers.__args_SilverballersArgs("SilverballersArgs(silverballers.__args)")
```
<!-- GRAPH ENDS HERE -->