// Copyright 2021-2023 Protocol Labs
// SPDX-License-Identifier: Apache-2.0, MIT

use std::io::Read;

use fil_actors_runtime::actor_dispatch;
use fil_actors_runtime::actor_error;
use fil_actors_runtime::builtin::singletons::SYSTEM_ACTOR_ADDR;
use fil_actors_runtime::runtime::{ActorCode, Runtime};
use fil_actors_runtime::ActorError;
use fvm_ipld_encoding::RawBytes;
use fvm_shared::sys::out;
use std::cmp;

use crate::{
    Method, PredictLinearRegressionParams, PredictLogisticRegressionParams,
    TrainLinearRegressionParams, TrainLogisticRegressionParams, MACHINELEARNING_ACTOR_NAME,
};

fil_actors_runtime::wasm_trampoline!(Actor);

fvm_sdk::sys::fvm_syscalls! {
    module = "mlsyscall_kernel";
    pub fn train_linear_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_linear_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      model_offset: u32,
      model_length: u32,
  ) -> Result<u32>;
  pub fn train_logistic_regression_syscall(
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
  ) -> Result<u32>;
  pub fn predict_logistic_regression_syscall(
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32>;
}

pub struct Actor;
impl Actor {
    fn train_linear_regression(
        rt: &impl Runtime,
        params: TrainLinearRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 95 + 9 * input_matrix[0].len();

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut result: Vec<u8> = vec![0; output_length];
            let value: u32 = train_linear_regression_syscall(
                data_offset,
                data_length,
                result.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            Ok(result.to_vec())
        }
    }

    fn predict_linear_regression(
        rt: &impl Runtime,
        params: PredictLinearRegressionParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1 + 2 * input_matrix.len();

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut result_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_linear_regression_syscall(
                data_offset,
                data_length,
                result_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_logistic_regression(
        rt: &impl Runtime,
        params: TrainLinearRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 172 + 9 * input_matrix[0].len();

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut result: Vec<u8> = vec![0; output_length];
            let value: u32 = train_logistic_regression_syscall(
                data_offset,
                data_length,
                result.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            Ok(result.to_vec())
        }
    }

    fn predict_logistic_regression(
        rt: &impl Runtime,
        params: PredictLinearRegressionParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1 + 1 * input_matrix.len();

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut result_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_logistic_regression_syscall(
                data_offset,
                data_length,
                result_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }
}

impl ActorCode for Actor {
    type Methods = Method;

    fn name() -> &'static str {
        MACHINELEARNING_ACTOR_NAME
    }

    actor_dispatch! {
      TrainLinearRegression => train_linear_regression,
      PredictLinearRegression => predict_linear_regression,
      TrainLogisticRegression => train_logistic_regression,
      PredictLogisticRegression => predict_logistic_regression,
    }
}
