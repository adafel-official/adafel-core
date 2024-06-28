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

use crate::{Method, TrainLinearRegressionParams, MACHINELEARNING_ACTOR_NAME};

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
}

pub struct Actor;
impl Actor {
    fn train_linear_regression(
        rt: &impl Runtime,
        params: TrainLinearRegressionParams,
    ) -> Result<Vec<u8>, ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let user_activity_matrix: Vec<Vec<i64>> = params.input_matrix;

            let conv_matrix: Vec<i64> = params.labels;

            let output_length = 95 + 9 * user_activity_matrix[0].len();

            let array = fvm_ipld_encoding::RawBytes::serialize(user_activity_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(conv_matrix).unwrap();

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

            println!("output is: {:?}", result);

            Ok(result.to_vec())
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
    }
}
