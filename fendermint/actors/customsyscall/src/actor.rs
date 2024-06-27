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
    fn invoke(rt: &impl Runtime) -> Result<[u8; 24], ActorError> {
        rt.validate_immediate_caller_accept_any()?;

        unsafe {
            let user_activity_matrix: Vec<Vec<i64>> = vec![
                vec![10900, 10000, 10000, 10000, 10000],
                vec![20600, 20000, 20000, 20200, 20200],
                vec![30000, 30600, 30700, 30100, 30000],
                vec![47000, 40800, 40700, 45000, 40600],
                vec![50000, 56000, 59005, 50200, 50000],
                vec![66000, 67000, 60050, 607600, 60000],
                vec![70000, 79000, 75000, 70800, 70000],
            ];

            let conv_matrix: Vec<i64> = vec![10000, 20000, 30000, 40000, 50000, 60000, 70000];

            let array = fvm_ipld_encoding::RawBytes::serialize(user_activity_matrix).unwrap();
            let conv_array = fvm_ipld_encoding::RawBytes::serialize(conv_matrix).unwrap();

            let data_offset = array.bytes().as_ptr() as u32;
            let data_length = array.bytes().len() as u32;
            let conv_offset = conv_array.bytes().as_ptr() as u32;
            let conv_length = conv_array.bytes().len() as u32;

            let mut result: [u8; 24] = [0; 24];
            let value: u32 = my_custom_syscall(
                data_offset,
                data_length,
                result.as_ptr() as u32,
                24 as u32,
                conv_offset,
                conv_length,
            )
            .unwrap();

            println!("output is: {:?}", result);

            Ok(result)
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
