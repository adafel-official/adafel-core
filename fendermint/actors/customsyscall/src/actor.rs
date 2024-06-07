// Copyright 2021-2023 Protocol Labs
// SPDX-License-Identifier: Apache-2.0, MIT

use std::io::Read;

use fil_actors_runtime::actor_dispatch;
use fil_actors_runtime::actor_error;
use fil_actors_runtime::builtin::singletons::SYSTEM_ACTOR_ADDR;
use fil_actors_runtime::runtime::{ActorCode, Runtime};
use fil_actors_runtime::ActorError;
use fvm_shared::sys::out;

use crate::{Method, CUSTOMSYSCALL_ACTOR_NAME};

fil_actors_runtime::wasm_trampoline!(Actor);

fvm_sdk::sys::fvm_syscalls! {
    module = "my_custom_kernel";
    pub fn my_custom_syscall(
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
    fn invoke(rt: &impl Runtime) -> Result<[u8; 9], ActorError> {
        rt.validate_immediate_caller_is(std::iter::once(&SYSTEM_ACTOR_ADDR))?;

        unsafe {
            let user_activity_matrix: Vec<Vec<i128>> = vec![
                vec![2, 2, 2, 2, 2],
                vec![2, 2, 2, 2, 2],
                vec![2, 2, 2, 2, 2],
                vec![2, 2, 2, 2, 2],
            ];

            let conv_matrix: Vec<Vec<i128>> = vec![vec![1, 0, 1], vec![1, 0, 1], vec![1, 0, 1]];

            let array = fvm_ipld_encoding::RawBytes::serialize(user_activity_matrix)?;
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(conv_matrix)?;

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut output: [u8; 9] = [0; 9];
            let value: u32 = my_custom_syscall(
                data_offset,
                data_length,
                output.as_ptr() as u32,
                9 as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            Ok(output)
        }
    }
}

impl ActorCode for Actor {
    type Methods = Method;

    fn name() -> &'static str {
        CUSTOMSYSCALL_ACTOR_NAME
    }

    actor_dispatch! {
        Invoke => invoke,
    }
}
