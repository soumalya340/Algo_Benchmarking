So i have to implement 4 algo's through Python  
1. Covariance matrix adaptation evolution strategy
3. Simulated Annealing
4. Differential evolution
5. Genetic algorithm

### For SouSwap.sol

```
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.9;

import "hardhat/console.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract SouSwap {
    IERC20 public token;

    uint public totalLiquidity;
    mapping(address => uint) public liquidityByUser;

    constructor(address token_addr) {
        token = IERC20(token_addr);
    }

    function init(uint _tokens) public payable returns (uint) {
        require(totalLiquidity == 0, "DEX already has liquidity");
        totalLiquidity = address(this).balance;
        liquidityByUser[msg.sender] = totalLiquidity;
        require(token.transferFrom(msg.sender, address(this), _tokens));
        return totalLiquidity;
    }

    // Trade Ether for tokens
    function ethToToken() public payable returns (uint) {
        uint token_reserve = token.balanceOf(address(this));
        uint currentBalance = address(this).balance - msg.value;
        uint tokens_bought = getInputPrice(
            msg.value,
            currentBalance,
            token_reserve
        );
        require(
            token.transfer(msg.sender, tokens_bought),
            "Token transfer failed"
        );
        return tokens_bought;
    }

    // Trade tokens for Ether
    function tokenToEth(uint _tokens) public returns (uint) {
        uint token_reserve = token.balanceOf(address(this));
        uint eth_bought = getInputPrice(
            _tokens,
            token_reserve,
            address(this).balance
        );
        require(
            token.transferFrom(msg.sender, address(this), _tokens),
            "Token transfer failed"
        );
        (bool success, ) = msg.sender.call{value: eth_bought}("");
        require(success, "ETH transfer failed");
        return eth_bought;
    }

    // Get the token price given input and reserves
    function getInputPrice(
        uint256 input_amount,
        uint256 input_reserve,
        uint256 output_reserve
    ) public pure returns (uint256) {
        require(input_reserve > 0 && output_reserve > 0, "INVALID_VALUE");
        uint input_amount_with_fee = input_amount * 997;
        uint numerator = input_amount_with_fee * output_reserve;
        uint denominator = (input_reserve * 1000) + input_amount_with_fee;
        return numerator / denominator;
    }

    // Deposit Ether and corresponding tokens into the DEX and receive liquidity tokens
    function provideLiquidity() public payable returns (uint256) {
        uint256 eth_reserve = address(this).balance - msg.value;
        uint256 token_reserve = token.balanceOf(address(this));
        uint256 token_amount = (msg.value * token_reserve) / eth_reserve;
        uint256 liquidity_minted = (msg.value * totalLiquidity) / eth_reserve;
        if (token.balanceOf(msg.sender) < token_amount) {
            (bool success, ) = msg.sender.call{value: msg.value}("");
            require(success, "ETH transfer failed");
        }
        liquidityByUser[msg.sender] += liquidity_minted;
        totalLiquidity += liquidity_minted;
        require(
            token.transferFrom(msg.sender, address(this), token_amount),
            "Token transfer failed"
        );
        return liquidity_minted;
    }

    // Withdraw liquidityByUser from the DEX, returning the user's proportional share of Ether and tokens
    function withdrawLiquidity(uint _amount) public returns (uint256, uint256) {
        require(
            liquidityByUser[msg.sender] >= _amount,
            "Insufficient liquidityByUser"
        );
        uint256 eth_reserve = address(this).balance;
        uint256 token_reserve = token.balanceOf(address(this));

        uint256 eth_amount = (_amount * eth_reserve) / totalLiquidity;
        uint256 token_amount = (_amount * token_reserve) / totalLiquidity;

        liquidityByUser[msg.sender] -= _amount;
        totalLiquidity -= _amount;

        (bool success, ) = msg.sender.call{value: eth_amount}("");
        require(success, "ETH transfer failed");
        require(
            token.transfer(msg.sender, token_amount),
            "Token transfer failed"
        );

        return (eth_amount, token_amount);
    }
}
```

