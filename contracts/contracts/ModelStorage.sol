//SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "./DataStorage.sol";

contract ModelStorage is Ownable {

  DataStorage dataStorage;
  uint256 public maxStake;

  struct Model {
    address contributor;
    bytes model;
    string name;
  }

  mapping(bytes32 => Model) public modelNameMap;
  mapping(address => uint256) public stakes;

  constructor(DataStorage _dataStorage) Ownable(msg.sender) {
    dataStorage = _dataStorage;
	}

  function addModel(bytes memory _model, string memory _name) external {
    modelNameMap[keccak256(abi.encodePacked(_name))] = Model(msg.sender, _model, _name);
  }

  function stakeAdafel(address staker) external payable {
    stakes[staker] += msg.value;
  }

  function withDrawStakes() external {
    
    payable(msg.sender).transfer(stakes[msg.sender]);

    stakes[msg.sender] = 0;
  }

  function getModel(string memory _modelname) external view returns (bytes memory) {
    
    if(modelNameMap[keccak256(abi.encodePacked(_modelname))].contributor == msg.sender) {
      return modelNameMap[keccak256(abi.encodePacked(_modelname))].model;
    }
    
    uint256 stakesRequired = maxStake - dataStorage.consumerCredits(msg.sender);

    require(stakes[msg.sender] >= stakesRequired, "insufficient stakes");

    return modelNameMap[keccak256(abi.encodePacked(_modelname))].model;
  
  }
}