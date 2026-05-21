use crate::{
    error::{TRTError, TRTResult},
    tensor::{Shape, Tensor},
};
use crate::DataType;
use cuda_rs::stream::CuStream;
use cuda_rs_sys::{
    cuGraphDestroy, cuGraphExecDestroy, cuGraphLaunch, cuStreamBeginCapture_v2,
    cuStreamEndCapture, CUgraph, CUgraphExec, CUgraphNode,
};
use tensorrt_rs_sys::{
    logger::Severity,
    runtime::{CudaEngine, ExecutionContext, Runtime},
};
use std::{collections::HashMap, fs, path::Path};

extern "C" {
    fn cuGraphInstantiate_v2(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        phErrorNode: *mut CUgraphNode,
        logBuffer: *mut std::os::raw::c_char,
        bufferSize: usize,
    ) -> std::os::raw::c_uint;
}

pub struct TRTEngine {
    runtime: Option<Runtime>,
    engine: Option<CudaEngine>,
    context: Option<ExecutionContext>,
    stream: CuStream,
    tensors: HashMap<String, Tensor>,
    cuda_graph_exec: Option<CUgraphExec>,
    graph_captured_shapes: HashMap<String, Shape>,
}

impl TRTEngine {
    pub fn new(engine_data: Vec<u8>, stream: &CuStream) -> TRTResult<Self> {

        let mut runtime = match Runtime::new() {
            Some(runtime) => runtime,
            None => return Err(TRTError::RuntimeCreationError),
        };

        let engine = match runtime.deserialize(engine_data.as_slice()) {
            Some(engine) => engine,
            None => return Err(TRTError::EngineDeserializationError),
        };

        Ok(Self {
            runtime: Some(runtime),
            engine: Some(engine),
            context: None,
            stream: stream.clone(),
            tensors: HashMap::new(),
            cuda_graph_exec: None,
            graph_captured_shapes: HashMap::new(),
        })
    }

    // TODO: reuse device memory
    pub fn activate(&mut self) -> TRTResult<()> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        self.context = match engine.create_execution_context() {
            Some(context) => Some(context),
            None => return Err(TRTError::ExecutionContextCreationError),
        };

        Ok(())
    }

    pub fn allocate_io_tensors(
        &mut self,
        max_shape_dict: &HashMap<&str, &Shape>,
        stream: Option<&CuStream>,
    ) -> TRTResult<()> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        let context: &mut ExecutionContext = match self.context.as_mut() {
            Some(context) => context,
            None => return Err(TRTError::ExecutionContextNotInitialized),
        };
        let stream = match stream {
            Some(stream) => stream,
            None => &self.stream,
        };

        let num_io_tensors = engine.get_num_io_tensors();

        for i in 0..num_io_tensors {
            let name = engine.get_io_tensor_name(i);
            let shape = engine.get_tensor_shape(name);
            let shape = Shape(shape);
            let shape = match max_shape_dict.get(name) {
                Some(max_shape) => max_shape,
                None => &shape,
            };
            if shape.0.iter().any(|&dim| dim < 0) {
                return Err(TRTError::ShapeError(shape.0.clone()));
            }
            if engine.get_tensor_io_mode(name).is_input() {
                if !context.set_input_shape(name, shape.0.as_slice()) {
                    return Err(TRTError::ShapeError(shape.0.clone()));
                }
            }

            let dtype = engine.get_tensor_dtype(name);
            let tensor = Tensor::empty(&shape, dtype, stream)?;
            let ptr = unsafe { tensor.get_raw_ptr() };
            self.tensors.insert(name.to_string(), tensor);
            if !context.set_tensor_address(name, ptr as _) {
                return Err(TRTError::InvalidAddress);
            }
        }

        // TODO: validate shapes, (batch size)

        Ok(())
    }

    pub fn capture_cuda_graph(&mut self) -> TRTResult<()> {
        let context = self
            .context
            .as_mut()
            .ok_or(TRTError::ExecutionContextNotInitialized)?;

        self.graph_captured_shapes.clear();
        for (name, tensor) in &self.tensors {
            self.graph_captured_shapes
                .insert(name.clone(), tensor.shape().clone());
        }

        let stream_raw = unsafe { self.stream.get_raw() };

        unsafe {
            let res = cuStreamBeginCapture_v2(stream_raw, 2);
            if res != 0 {
                return Err(TRTError::CudaError(cuda_rs::error::CuError::from(res)));
            }
        }

        if !context.enqueue_v3(&self.stream) {
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe {
                cuStreamEndCapture(stream_raw, &mut graph);
                if !graph.is_null() {
                    cuGraphDestroy(graph);
                }
            }
            return Err(TRTError::EnqueueError);
        }

        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let res = cuStreamEndCapture(stream_raw, &mut graph);
            if res != 0 {
                return Err(TRTError::CudaError(cuda_rs::error::CuError::from(res)));
            }
        }

        let mut exec_graph: CUgraphExec = std::ptr::null_mut();
        unsafe {
            let res = cuGraphInstantiate_v2(
                &mut exec_graph,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            cuGraphDestroy(graph);
            if res != 0 {
                return Err(TRTError::CudaError(cuda_rs::error::CuError::from(res)));
            }
        }

        if let Some(old_exec) = self.cuda_graph_exec.replace(exec_graph) {
            unsafe { cuGraphExecDestroy(old_exec) };
        }

        Ok(())
    }

    pub fn invalidate_cuda_graph(&mut self) {
        if let Some(exec) = self.cuda_graph_exec.take() {
            unsafe { cuGraphExecDestroy(exec) };
        }
        self.graph_captured_shapes.clear();
    }

    pub fn inference(
        &mut self,
        feed_dict: &HashMap<&str, &Tensor>,
        stream_opt: Option<&CuStream>,
    ) -> TRTResult<&HashMap<String, Tensor>> {
        if stream_opt.is_some() {
            self.invalidate_cuda_graph();
        }

        let stream = match stream_opt {
            Some(s) => s,
            None => &self.stream,
        };

        let context = self.context.as_mut()
            .ok_or(TRTError::ExecutionContextNotInitialized)?;

        for (name, input_tensor) in feed_dict {
            let tensor = match self.tensors.get_mut(name.to_owned()) {
                Some(t) => t,
                None => continue,
            };
            let new_shape = input_tensor.shape();
            if tensor.shape() != new_shape {
                unsafe { tensor.reset_shape(new_shape)? };
                if !context.set_input_shape(name, new_shape.0.as_slice()) {
                    return Err(TRTError::ShapeError(new_shape.0.clone()));
                }
                if let Some(exec) = self.cuda_graph_exec.take() {
                    unsafe { cuGraphExecDestroy(exec); }
                }
                self.graph_captured_shapes.clear();
            }
            tensor.copy_from(input_tensor, Some(stream))?;
        }

        if let Some(exec_graph) = self.cuda_graph_exec {
            let stream_raw = unsafe { self.stream.get_raw() };
            unsafe {
                let res = cuGraphLaunch(exec_graph, stream_raw);
                if res != 0 {
                    return Err(TRTError::CudaError(cuda_rs::error::CuError::from(
                        res,
                    )));
                }
            }
            return Ok(&self.tensors);
        }

        if !context.enqueue_v3(stream) {
            return Err(TRTError::EnqueueError);
        }

        Ok(&self.tensors)
    }

    pub fn log(&mut self, level: Severity, msg: &str) {
        self.runtime.as_mut().unwrap().logger().log(level, msg);
    }

    pub fn get_tensor_shape(&self, name: &str) -> Option<Vec<i32>> {
        self.engine.as_ref().map(|e| e.get_tensor_shape(name))
    }

    pub fn get_tensor_dtype(&self, name: &str) -> Option<DataType> {
        self.engine.as_ref().map(|e| e.get_tensor_dtype(name))
    }

    pub fn io_tensor_names(&self) -> Vec<(String, bool)> {
        self.engine.as_ref().map_or(Vec::new(), |e| {
            let count = e.get_num_io_tensors();
            (0..count)
                .map(|i| {
                    let name = e.get_io_tensor_name(i).to_string();
                    let is_input = e.get_tensor_io_mode(&name).is_input();
                    (name, is_input)
                })
                .collect()
        })
    }

    pub fn enumerate_tensors(&self) -> Vec<(String, Vec<i32>, DataType, bool)> {
        self.engine.as_ref().map_or(Vec::new(), |e| {
            let count = e.get_num_io_tensors();
            (0..count)
                .map(|i| {
                    let name = e.get_io_tensor_name(i).to_string();
                    let shape = e.get_tensor_shape(&name);
                    let dtype = e.get_tensor_dtype(&name);
                    let is_input = e.get_tensor_io_mode(&name).is_input();
                    (name, shape, dtype, is_input)
                })
                .collect()
        })
    }

    pub fn allocate_io_tensors_float(&mut self, stream: &CuStream) -> TRTResult<()> {
        let tensors = self.enumerate_tensors();
        let context = self.context.as_mut()
            .ok_or(TRTError::ExecutionContextNotInitialized)?;

        for (name, shape, _dtype, is_input) in &tensors {
            let shape = Shape(shape.clone());
            let tensor = Tensor::empty(&shape, DataType::FLOAT, stream)?;
            let ptr = unsafe { tensor.get_raw_ptr() };
            if !context.set_tensor_address(name, ptr as _) {
                return Err(TRTError::InvalidAddress);
            }
            if *is_input {
                if !context.set_input_shape(name, &shape.0) {
                    return Err(TRTError::ShapeError(shape.0.clone()));
                }
            }
            self.tensors.insert(name.clone(), tensor);
        }
        Ok(())
    }
}

impl Drop for TRTEngine {
    fn drop(&mut self) {
        self.invalidate_cuda_graph();

        if let Some(context) = self.context.take() {
            std::mem::drop(context);
        }

        if let Some(engine) = self.engine.take() {
            std::mem::drop(engine);
        }

        if let Some(runtime) = self.runtime.take() {
            std::mem::drop(runtime);
        }
    }
}
