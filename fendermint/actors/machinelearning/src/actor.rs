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
    Method, PredictDecisionTreeClassificationParams, PredictDecisionTreeRegressionParams,
    PredictKNNClassificationParams, PredictKNNRegressionParams, PredictLinearRegressionParams,
    PredictLogisticRegressionParams, PredictRandomForestClassificationParams,
    PredictRandomForestRegressionParams, TrainDecisionTreeClassificationParams,
    TrainDecisionTreeRegressionParams, TrainKNNClassificationParams, TrainKNNRegressionParams,
    TrainLinearRegressionParams, TrainLogisticRegressionParams,
    TrainRandomForestClassificationParams, TrainRandomForestRegressionParams,
    MACHINELEARNING_ACTOR_NAME,
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
    pub fn train_knn_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_knn_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      model_offset: u32,
      model_length: u32,
    ) -> Result<u32>;
    pub fn train_knn_classification_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_knn_classification_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      model_offset: u32,
      model_length: u32,
    ) -> Result<u32>;
    pub fn train_decision_tree_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_decision_tree_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      model_offset: u32,
      model_length: u32,
    ) -> Result<u32>;
    pub fn train_decision_tree_classification_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_decision_tree_classification_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      model_offset: u32,
      model_length: u32,
    ) -> Result<u32>;
    pub fn train_random_forest_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_random_forest_regression_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      model_offset: u32,
      model_length: u32,
    ) -> Result<u32>;
    pub fn train_random_forest_classification_syscall(
      data_offset: u32,
      data_length: u32,
      output_offset: u32,
      output_length: u32,
      conv_offset: u32,
      conv_length: u32,
    ) -> Result<u32>;
    pub fn predict_random_forest_classification_syscall(
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

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_linear_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

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

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_linear_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_logistic_regression(
        rt: &impl Runtime,
        params: TrainLogisticRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_logistic_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

            Ok(result.to_vec())
        }
    }

    fn predict_logistic_regression(
        rt: &impl Runtime,
        params: PredictLogisticRegressionParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_logistic_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_knn_regression(
        rt: &impl Runtime,
        params: TrainKNNRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_knn_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

            Ok(result.to_vec())
        }
    }

    fn predict_knn_regression(
        rt: &impl Runtime,
        params: PredictKNNRegressionParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_knn_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_knn_classification(
        rt: &impl Runtime,
        params: TrainKNNClassificationParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_knn_classification_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

            Ok(result.to_vec())
        }
    }

    fn predict_knn_classification(
        rt: &impl Runtime,
        params: PredictKNNClassificationParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_knn_classification_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_decision_tree_regression(
        rt: &impl Runtime,
        params: TrainDecisionTreeRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_decision_tree_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

            Ok(result.to_vec())
        }
    }

    fn predict_decision_tree_regression(
        rt: &impl Runtime,
        params: PredictDecisionTreeRegressionParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_decision_tree_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_decision_tree_classification(
        rt: &impl Runtime,
        params: TrainDecisionTreeClassificationParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_decision_tree_classification_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

            Ok(result.to_vec())
        }
    }

    fn predict_decision_tree_classification(
        rt: &impl Runtime,
        params: PredictDecisionTreeClassificationParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_decision_tree_classification_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }
    fn train_random_forest_regression(
        rt: &impl Runtime,
        params: TrainRandomForestRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_random_forest_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);
            Ok(result.to_vec())
        }
    }

    fn predict_random_forest_regression(
        rt: &impl Runtime,
        params: PredictRandomForestRegressionParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_random_forest_regression_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

            let result: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
                &fvm_ipld_encoding::RawBytes::new(result_raw),
            )
            .unwrap();

            Ok(result)
        }
    }

    fn train_random_forest_classification(
        rt: &impl Runtime,
        params: TrainRandomForestClassificationParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(params.labels).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = train_random_forest_classification_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            let mut result: Vec<u8> = vec![0; value as usize];
            result.copy_from_slice(&output_raw[..(value as usize)]);

            Ok(result.to_vec())
        }
    }

    fn predict_random_forest_classification(
        rt: &impl Runtime,
        params: PredictRandomForestClassificationParams,
    ) -> Result<Vec<i64>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let input_matrix: Vec<Vec<i64>> = params.input_matrix;

            let output_length = 1000000;

            let array = fvm_ipld_encoding::RawBytes::serialize(input_matrix).unwrap();
            let model_array = fvm_ipld_encoding::RawBytes::serialize(params.model).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let model_offset = model_array.bytes().as_ptr() as u32;
            let model_length = model_array.bytes().len() as u32;

            let mut output_raw: Vec<u8> = vec![0; output_length];
            let value: u32 = predict_random_forest_classification_syscall(
                data_offset,
                data_length,
                output_raw.as_ptr() as u32,
                output_length as u32,
                model_offset,
                model_length,
            )
            .unwrap();

            let mut result_raw: Vec<u8> = vec![0; value as usize];
            result_raw.copy_from_slice(&output_raw[..(value as usize)]);

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
      TrainKNNRegression => train_knn_regression,
      PredictKNNRegression => predict_knn_regression,
      TrainKNNClassification => train_knn_classification,
      PredictKNNClassification => predict_knn_classification,
      TrainDecisionTreeRegression => train_decision_tree_regression,
      PredictDecisionTreeRegression => predict_decision_tree_regression,
      TrainDecisionTreeClassification => train_decision_tree_classification,
      PredictDecisionTreeClassification => predict_decision_tree_classification,
      TrainRandomForestRegression => train_random_forest_regression,
      PredictRandomForestRegression => predict_random_forest_regression,
      TrainRandomForestClassification => train_random_forest_classification,
      PredictRandomForestClassification => predict_random_forest_classification,
    }
}
