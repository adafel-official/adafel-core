// Copyright 2022-2023 Protocol Labs
// SPDX-License-Identifier: MIT
//! Join subnet cli command handler.

use async_trait::async_trait;
use clap::Args;
use ipc_sdk::subnet_id::SubnetID;
use std::{fmt::Debug, str::FromStr};

use crate::{
    f64_to_token_amount, get_ipc_provider, require_fil_addr_from_str, CommandLineHandler,
    GlobalArguments,
};

/// The command to join a subnet
pub struct JoinSubnet;

#[async_trait]
impl CommandLineHandler for JoinSubnet {
    type Arguments = JoinSubnetArgs;

    async fn handle(global: &GlobalArguments, arguments: &Self::Arguments) -> anyhow::Result<()> {
        log::debug!("join subnet with args: {:?}", arguments);

        let mut provider = get_ipc_provider(global)?;
        let subnet = SubnetID::from_str(&arguments.subnet)?;
        let from = match &arguments.from {
            Some(address) => Some(require_fil_addr_from_str(address)?),
            None => None,
        };
        let public_key = hex::decode(&arguments.public_key)?;
        if let Some(initial_balance) = arguments.initial_balance {
            println!("pre-funding address with {initial_balance}");
            provider
                .prefund_subnet(subnet.clone(), from, f64_to_token_amount(initial_balance)?)
                .await?;
        }
        let epoch = provider
            .join_subnet(
                subnet,
                from,
                f64_to_token_amount(arguments.collateral)?,
                public_key,
            )
            .await?;
        println!("joined at epoch: {epoch}");

        Ok(())
    }
}

#[derive(Debug, Args)]
#[command(name = "join", about = "Join a subnet")]
pub struct JoinSubnetArgs {
    #[arg(long, short, help = "The address that joins the subnet")]
    pub from: Option<String>,
    #[arg(long, short, help = "The subnet to join")]
    pub subnet: String,
    #[arg(
        long,
        short,
        help = "The collateral to stake in the subnet (in whole FIL units)"
    )]
    pub collateral: f64,
    #[arg(long, short, help = "The validator's metadata, hex encoded")]
    pub public_key: String,
    #[arg(
        long,
        help = "Optionally add an initial balance to the validator in genesis in the subnet"
    )]
    pub initial_balance: Option<f64>,
}

/// The command to stake in a subnet from validator
pub struct StakeSubnet;

#[async_trait]
impl CommandLineHandler for StakeSubnet {
    type Arguments = StakeSubnetArgs;

    async fn handle(global: &GlobalArguments, arguments: &Self::Arguments) -> anyhow::Result<()> {
        log::debug!("join subnet with args: {:?}", arguments);

        let mut provider = get_ipc_provider(global)?;
        let subnet = SubnetID::from_str(&arguments.subnet)?;
        let from = match &arguments.from {
            Some(address) => Some(require_fil_addr_from_str(address)?),
            None => None,
        };
        provider
            .stake(subnet, from, f64_to_token_amount(arguments.collateral)?)
            .await
    }
}

#[derive(Debug, Args)]
#[command(name = "stake", about = "Add collateral to an already joined subnet")]
pub struct StakeSubnetArgs {
    #[arg(long, short, help = "The address that stakes in the subnet")]
    pub from: Option<String>,
    #[arg(long, short, help = "The subnet to add collateral to")]
    pub subnet: String,
    #[arg(
        long,
        short,
        help = "The collateral to stake in the subnet (in whole FIL units)"
    )]
    pub collateral: f64,
}

/// The command to unstake in a subnet from validator
pub struct UnstakeSubnet;

#[async_trait]
impl CommandLineHandler for UnstakeSubnet {
    type Arguments = UnstakeSubnetArgs;

    async fn handle(global: &GlobalArguments, arguments: &Self::Arguments) -> anyhow::Result<()> {
        log::debug!("join subnet with args: {:?}", arguments);

        let mut provider = get_ipc_provider(global)?;
        let subnet = SubnetID::from_str(&arguments.subnet)?;
        let from = match &arguments.from {
            Some(address) => Some(require_fil_addr_from_str(address)?),
            None => None,
        };
        provider
            .unstake(subnet, from, f64_to_token_amount(arguments.collateral)?)
            .await
    }
}

#[derive(Debug, Args)]
#[command(
    name = "unstake",
    about = "Remove collateral to an already joined subnet"
)]
pub struct UnstakeSubnetArgs {
    #[arg(long, short, help = "The address that unstakes in the subnet")]
    pub from: Option<String>,
    #[arg(long, short, help = "The subnet to release collateral from")]
    pub subnet: String,
    #[arg(
        long,
        short,
        help = "The collateral to unstake from the subnet (in whole FIL units)"
    )]
    pub collateral: f64,
}

pub struct PreFundSubnet;

#[async_trait]
impl CommandLineHandler for PreFundSubnet {
    type Arguments = PreFundSubnetArgs;

    async fn handle(global: &GlobalArguments, arguments: &Self::Arguments) -> anyhow::Result<()> {
        log::debug!("pre-fund subnet with args: {:?}", arguments);

        let mut provider = get_ipc_provider(global)?;
        let subnet = SubnetID::from_str(&arguments.subnet)?;
        let from = match &arguments.from {
            Some(address) => Some(require_fil_addr_from_str(address)?),
            None => None,
        };
        provider
            .prefund_subnet(
                subnet.clone(),
                from,
                f64_to_token_amount(arguments.initial_balance)?,
            )
            .await?;
        println!("address pre-funded successfully");

        Ok(())
    }
}

#[derive(Debug, Args)]
#[command(
    name = "pre-fund",
    about = "Add some funds in genesis to an address in a child-subnet"
)]
pub struct PreFundSubnetArgs {
    #[arg(long, short, help = "The address funded in the subnet")]
    pub from: Option<String>,
    #[arg(long, short, help = "The subnet to add balance to")]
    pub subnet: String,
    #[arg(help = "Add an initial balance for the address in genesis in the subnet")]
    pub initial_balance: f64,
}
