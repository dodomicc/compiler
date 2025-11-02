"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.orderCreateHandler = void 0;
const order_1 = require("../OrderDataStructure/order");
const executedResult_1 = require("../Type/executedResult");
const orderCreateHandler = (order, orderBook, tradingPair) => {
    // this is to calculate the amount need for a market order, thus can be used to freeze the amount before executing a market order
    const requiredlAmount = order.type == order_1.OrderType.MARKET ? orderBook.calculateMarketPrice(order.side, order.size).price : undefined;
    // executed the order
    const orderCreateHandlerResult = orderBook.createOrder(order);
    //------------------------------------------------------------------------------------------------------------------------- 
    // add the executed result into the output result, success or failure
    const executedResult = orderCreateHandlerResult.err ? executedResult_1.ExecutedResult.FAILURE : executedResult_1.ExecutedResult.SUCCESS;
    // add the current orderInfo into the order
    const createOrderInfo = { id: order.orderID, type: order.type, size: order.size, quantityLeft: orderCreateHandlerResult.quantityLeft };
    // add the done orders info into the result
    const doneOrders = orderCreateHandlerResult.done;
    const doneOrderIds = [];
    doneOrders.map((order) => { doneOrderIds.push(order.id); });
    // add the partial processed order info into result
    const partialProcessedOrderId = orderCreateHandlerResult.partial ? orderCreateHandlerResult.partial.id : undefined;
    const partialProcessedOrderRemainingSize = orderCreateHandlerResult.partial ? orderCreateHandlerResult.partial.size.toNumber() : undefined;
    // calculate the current price and the depth of the order book
    const dealVolume = order.size - orderCreateHandlerResult.quantityLeft;
    let partialQuantityProcesse = dealVolume;
    let dealTotalVal;
    // console.log(orderCreateHandlerResult.partialQuantityProcessed)
    if (order.type == order_1.OrderType.MARKET) {
        dealTotalVal = requiredlAmount;
    }
    else {
        dealTotalVal = 0;
        orderCreateHandlerResult.done.map((order) => {
            partialQuantityProcesse -= order.size.toNumber();
            dealTotalVal += order.size.toNumber() * order.price;
        });
        if (orderCreateHandlerResult.partial) {
            dealTotalVal += partialQuantityProcesse * orderCreateHandlerResult.partial.price;
        }
    }
    const currentPricePerUnit = dealTotalVal > 0 ? dealTotalVal / dealVolume : undefined;
    const depth = orderBook.depth();
    // make all result into the output result
    const outputResult = { executedResult, createOrderInfo, doneOrderIds, partialProcessedOrderId, partialProcessedOrderRemainingSize, currentPricePerUnit, depth };
    // return the result
    return outputResult;
};
exports.orderCreateHandler = orderCreateHandler;
const getSecondPairOfTradingPair = (tradingPair) => {
    return tradingPair.match(/.*\/(.*)/)[1];
};
const freezeAmount = (orderID, amount) => {
};
