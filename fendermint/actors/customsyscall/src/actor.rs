// Copyright 2021-2023 Protocol Labs
// SPDX-License-Identifier: Apache-2.0, MIT

use fil_actors_runtime::actor_dispatch;
use fil_actors_runtime::actor_error;
use fil_actors_runtime::builtin::singletons::SYSTEM_ACTOR_ADDR;
use fil_actors_runtime::runtime::{ActorCode, Runtime};
use fil_actors_runtime::ActorError;

use crate::InvokeParams;
use crate::{Method, CUSTOMSYSCALL_ACTOR_NAME};

fil_actors_runtime::wasm_trampoline!(Actor);

fvm_sdk::sys::fvm_syscalls! {
    module = "my_custom_kernel";
    pub fn my_custom_syscall(
        i1: u64,
        i2: u64,
        i3: u64,
        // i4: u64,
        // i5: u64,
        // i6: u64,
        // i7: u64,
        // i8: u64,
        // i9: u64,
        c1: u64,
        c2: u64,
        c3: u64,
        // c4: u64,
        // c5: u64,
        // c6: u64,
        // c7: u64,
        // c8: u64,
        // c9: u64,
    ) -> Result<u64>;
}

pub struct Actor;
impl Actor {
    fn invoke(rt: &impl Runtime, params: InvokeParams) -> Result<u64, ActorError> {
        rt.validate_immediate_caller_is(std::iter::once(&SYSTEM_ACTOR_ADDR))?;

        unsafe {
            let value = my_custom_syscall(
                params.i1, params.i2, params.i3,
                // params.i4, params.i5, params.i6, params.i7,
                // params.i8, params.i9,
                params.c1, params.c2,
                params.c3,
                // params.c4, params.c5,
                // params.c6, params.c7, params.c8, params.c9,
            )
            .unwrap();
            Ok(value)
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
