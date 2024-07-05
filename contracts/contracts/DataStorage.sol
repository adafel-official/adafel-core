//SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract DataStorage is Ownable {
	uint256[][] public userActivityMatrix;
	mapping(address => uint256) public addressToId;
	mapping(uint256 => address) public idToAddress;
	mapping(address => uint256) public consumerCredits;
	mapping(bytes32 => uint256) public schemaIndex;
	uint256 public latestIndex;
	uint256 public totalCategories;
  uint256 public userRewardPerDatapoint;

	enum Category {
		Gaming,
		Marketplace,
		Defi,
		Dao,
		Web3Social,
		Identity,
		Certificates
	}

  struct SchemaDetails {
    bytes32 schemaName;
		bytes32[] columns;
		Category schemaCategory;
    uint256 totalRecords;
  }

	struct Analytics {
		bytes32 schemaName;
		bytes32[] columns;
		Category schemaCategory;
		int64[][] data;
		mapping(address => uint256) addressToId;
		mapping(uint256 => address) idToAddress;
		mapping(bytes32 => uint256) columnToIndex;
	}

	Analytics[] public dappAnalytics;

	constructor() Ownable(msg.sender) {
		uint256[] memory initialMatrix;
		userActivityMatrix.push(initialMatrix);
		latestIndex = 0;
		totalCategories = 7;
		dappAnalytics.push();
    userRewardPerDatapoint = 10000000000000000;
	}

	event NewAnalytics(address user, address provider, uint256 category);

  event NewBatchAnalytics(address[] user, address provider, uint256 category);

	function addUser(address userAddress) external {
		// get the total length of current activity matrix
		latestIndex = latestIndex + 1;
		uint256[] memory initialMatrix;
		userActivityMatrix.push(initialMatrix);

		// add the new user details
		for (uint256 i = 0; i < totalCategories; i++) {
			userActivityMatrix[latestIndex].push(0);
		}

		// add user id to address mapping
		addressToId[userAddress] = latestIndex;
		idToAddress[latestIndex] = userAddress;
	}

	function addSchema(
		bytes32 schemaName,
		bytes32[] calldata columns,
		Category category
	) external {
		// Cannot have two schema with same name
    require(schemaIndex[schemaName] == uint256(0), "SCHEMA NAME EXISTS");

    // initializing schema with defaults
		Analytics storage analytics = dappAnalytics.push();
		analytics.schemaName = schemaName;
		analytics.schemaCategory = category;
		
    int64[] memory initialUser;
		analytics.data.push(initialUser);
		
    for (uint256 i = 0; i < columns.length; i++) {
			analytics.data[0].push(0);
			analytics.columns.push(columns[i]);
			analytics.columnToIndex[columns[i]] = i;
		}

		// adding to schema index map
		schemaIndex[schemaName] = dappAnalytics.length - 1;
	}

	function addAnalytics(
		address payable userAddress,
		bytes32 schemaName,
		bytes32[] calldata columns,
		int64[] calldata data
	) external payable {
		require(schemaIndex[schemaName] != 0, "SCHEMA NOT PRESENT");

		// add user if not already present
		if (addressToId[userAddress] == 0) {
			this.addUser(userAddress);
		}

		// retrieve storage instance
		Analytics storage schemaAnalytics = dappAnalytics[
			schemaIndex[schemaName]
		];

		// push new user if not already present
		if (schemaAnalytics.addressToId[userAddress] == 0) {
			schemaAnalytics.data.push();
			for (uint256 i = 0; i < schemaAnalytics.columns.length; i++) {
				schemaAnalytics.data[schemaAnalytics.data.length - 1].push(0);
			}

			schemaAnalytics.addressToId[userAddress] =
				schemaAnalytics.data.length -
				1;
			schemaAnalytics.idToAddress[
				schemaAnalytics.data.length - 1
			] = userAddress;
		}

		// add to the existing data
		for (uint256 i = 0; i < columns.length; i++) {
			schemaAnalytics.data[schemaAnalytics.addressToId[userAddress]][
				schemaAnalytics.columnToIndex[columns[i]]
			] += data[i];
		}

		userActivityMatrix[addressToId[userAddress]][
			uint256(schemaAnalytics.schemaCategory)
		] += 1;

		// rewarding the users for sharing data
		bool sent = userAddress.send(userRewardPerDatapoint);
		require(sent, "Failed to reward user");

		// increasing credit limit for provider
		consumerCredits[msg.sender] = consumerCredits[msg.sender] + 1;

		emit NewAnalytics(
			userAddress,
			msg.sender,
			uint256(schemaAnalytics.schemaCategory)
		);
	}

  function addBatchAnalytics(
		address[] memory userAddresses,
		bytes32 schemaName,
		bytes32[] calldata columns,
		int64[][] calldata data
	) external payable {
		require(schemaIndex[schemaName] != 0, "SCHEMA NOT PRESENT");

    // retrieve storage instance
		Analytics storage schemaAnalytics = dappAnalytics[
			schemaIndex[schemaName]
		];

    for (uint256 user = 0; user < userAddresses.length; user++) {
      // add user if not already present
      if (addressToId[userAddresses[user]] == 0) {
        this.addUser(userAddresses[user]);
      }

      // push new user if not already present
      if (schemaAnalytics.addressToId[userAddresses[user]] == 0) {
        schemaAnalytics.data.push();
        for (uint256 i = 0; i < schemaAnalytics.columns.length; i++) {
          schemaAnalytics.data[schemaAnalytics.data.length - 1].push(0);
        }

        schemaAnalytics.addressToId[userAddresses[user]] =
          schemaAnalytics.data.length -
          1;
        schemaAnalytics.idToAddress[
          schemaAnalytics.data.length - 1
        ] = userAddresses[user];
      }

      // add to the existing data
      for (uint256 i = 0; i < columns.length; i++) {
        schemaAnalytics.data[schemaAnalytics.addressToId[userAddresses[user]]][
          schemaAnalytics.columnToIndex[columns[i]]
        ] += data[user][i];
      }

      userActivityMatrix[addressToId[userAddresses[user]]][
        uint256(schemaAnalytics.schemaCategory)
      ] += 1;

      // rewarding the users for sharing data
      bool sent = payable(userAddresses[user]).send(userRewardPerDatapoint);
      require(sent, "Failed to reward user");
    }

    // increasing credit limit for provider
		consumerCredits[msg.sender] = consumerCredits[msg.sender] + userAddresses.length;

		emit NewBatchAnalytics(
			userAddresses,
			msg.sender,
			uint256(schemaAnalytics.schemaCategory)
		);
	}

	function updateAnalytics(
		address payable userAddress,
		bytes32 schemaName,
		bytes32[] calldata columns,
		int64[] calldata data
	) external payable {
		// add user if not already present
		if (addressToId[userAddress] == 0) {
			this.addUser(userAddress);
		}

		// retrieve storage instance
		Analytics storage schemaAnalytics = dappAnalytics[
			schemaIndex[schemaName]
		];

		// push new user if not already present
		if (schemaAnalytics.addressToId[userAddress] == 0) {
			schemaAnalytics.data.push();
			for (uint256 i = 0; i < schemaAnalytics.columns.length; i++) {
				schemaAnalytics.data[schemaAnalytics.data.length - 1].push(0);
			}

			schemaAnalytics.addressToId[userAddress] =
				schemaAnalytics.data.length -
				1;
			schemaAnalytics.idToAddress[
				schemaAnalytics.data.length - 1
			] = userAddress;
		}

		// replace the existing user data with new one
		for (uint256 i = 0; i < columns.length; i++) {
			schemaAnalytics.data[schemaAnalytics.addressToId[userAddress]][
				schemaAnalytics.columnToIndex[columns[i]]
			] = data[i];
		}

		userActivityMatrix[addressToId[userAddress]][
			uint256(schemaAnalytics.schemaCategory)
		] += 1;

		// rewarding the users for sharing data
		bool sent = userAddress.send(userRewardPerDatapoint);
		require(sent, "Failed to reward user");

		// increasing credit limit for provider
		consumerCredits[msg.sender] = consumerCredits[msg.sender] + 1;

		emit NewAnalytics(
			userAddress,
			msg.sender,
			uint256(schemaAnalytics.schemaCategory)
		);
	}

  function updateUserReward(uint256 newReward) external onlyOwner {
    userRewardPerDatapoint = newReward;
  }

	function getUserActivityMatrix()
		external
		view
		returns (uint256[][] memory)
	{
		return userActivityMatrix;
	}

  function getAllSchemas() external view returns (SchemaDetails[] memory) {
    SchemaDetails[] memory schemaDetails = new SchemaDetails[](dappAnalytics.length - 1);
    for (uint256 i = 1; i < dappAnalytics.length; i++) {
      // bytes32[] memory schemaColumns = new bytes32[](dappAnalytics[i].columns.length);
      SchemaDetails memory schemaDetail;
      schemaDetail.schemaName = dappAnalytics[i].schemaName;
      schemaDetail.columns = dappAnalytics[i].columns;
      schemaDetail.schemaCategory = dappAnalytics[i].schemaCategory;
      schemaDetail.totalRecords = dappAnalytics[i].data.length;
      schemaDetails[i - 1] = schemaDetail;
    }

    return schemaDetails;
  }

	function getAnalyticsDataBySchemaName(
		bytes32 schemaName
	) external view returns (int64[][] memory) {
		return dappAnalytics[schemaIndex[schemaName]].data;
	}

	function getColumnsOfSchema(
		bytes32 schemaName
	) external view returns (bytes32[] memory) {
		return dappAnalytics[schemaIndex[schemaName]].columns;
	}

  function getSchemaAddressToId(
    bytes32 schemaName,
    address userAddress
  ) external view returns (uint256) {
    return dappAnalytics[schemaIndex[schemaName]].addressToId[userAddress];
  }

  function getSchemaIdToAddress(
    bytes32 schemaName,
    uint256 userId
  ) external view returns (address) {
    return dappAnalytics[schemaIndex[schemaName]].idToAddress[userId];
  }

	receive() external payable {}
}
