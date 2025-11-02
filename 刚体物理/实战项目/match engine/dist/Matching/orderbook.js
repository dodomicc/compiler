"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrderBook = void 0;
const bignumber_js_1 = __importDefault(require("bignumber.js"));
const errors_1 = require("../ErrorHandling/errors");
const order_1 = require("../OrderDataStructure/order");
const orderside_1 = require("../OrderDataStructure/orderside");
const side_1 = require("../Type/side");
const validTimeInForce = Object.values(order_1.TimeInForce);
class OrderBook {
    constructor() {
        this.orders = {};
        /**
         *  Create a trade order
         *  @see {@link IProcessOrder} for the returned data structure
         *
         *  @param type - `limit` or `market`
         *  @param side - `sell` or `buy`
         *  @param size - How much of currency you want to trade in units of base currency
         *  @param price - The price at which the order is to be fullfilled, in units of the quote currency. Param only for limit order
         *  @param orderID - Unique order ID. Param only for limit order
         *  @param timeInForce - Time-in-force supported are: `GTC` (default), `FOK`, `IOC`. Param only for limit order
         *  @returns An object with the result of the processed order or an error.
         */
        this.createOrder = (createOrderType) => {
            switch (createOrderType.type) {
                case order_1.OrderType.MARKET:
                    return this.market(createOrderType.side, createOrderType.size);
                case order_1.OrderType.LIMIT:
                    return this.limit(createOrderType.side, createOrderType.orderID, createOrderType.size, createOrderType.price, createOrderType.timeInForce ? createOrderType.timeInForce : order_1.TimeInForce.GTC);
                default:
                    return {
                        done: [],
                        partial: null,
                        partialQuantityProcessed: 0,
                        quantityLeft: createOrderType.size,
                        err: (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidOrderType)
                    };
            }
        };
        /**
         * Create a market order
         *  @see {@link IProcessOrder} for the returned data structure
         *
         * @param side - `sell` or `buy`
         * @param size - How much of currency you want to trade in units of base currency
         * @returns An object with the result of the processed order or an error
         */
        this.market = (side, size) => {
            const response = {
                done: [],
                partial: null,
                partialQuantityProcessed: 0,
                quantityLeft: size,
                err: null
            };
            if (side !== side_1.Side.SELL && side !== side_1.Side.BUY) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidSide);
                return response;
            }
            if (typeof size !== 'number' || size <= 0) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInsufficientQuantity);
                return response;
            }
            let iter;
            let sideToProcess;
            if (side === side_1.Side.BUY) {
                iter = this.asks.minPriceQueue;
                sideToProcess = this.asks;
            }
            else {
                iter = this.bids.maxPriceQueue;
                sideToProcess = this.bids;
            }
            while (size > 0 && sideToProcess.len() > 0) {
                // if sideToProcess.len > 0 it is not necessary to verify that bestPrice exists
                const bestPrice = iter();
                const { done, partial, partialQuantityProcessed, quantityLeft } = this.processQueue(bestPrice, size);
                response.done = response.done.concat(done);
                response.partial = partial;
                response.partialQuantityProcessed = partialQuantityProcessed;
                size = quantityLeft;
            }
            response.quantityLeft = size;
            return response;
        };
        /**
         * Create a limit order
         *  @see {@link IProcessOrder} for the returned data structure
         *
         * @param side - `sell` or `buy`
         * @param orderID - Unique order ID
         * @param size - How much of currency you want to trade in units of base currency
         * @param price - The price at which the order is to be fullfilled, in units of the quote currency
         * @param timeInForce - Time-in-force type supported are: GTC, FOK, IOC
         * @returns An object with the result of the processed order or an error
         */
        this.limit = (side, orderID, size, price, timeInForce = order_1.TimeInForce.GTC) => {
            const response = {
                done: [],
                partial: null,
                partialQuantityProcessed: 0,
                quantityLeft: size,
                err: null
            };
            if (side !== side_1.Side.SELL && side !== side_1.Side.BUY) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidSide);
                return response;
            }
            if (this.orders[orderID] !== undefined) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrOrderExists);
                return response;
            }
            if (typeof size !== 'number' || size <= 0) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidQuantity);
                return response;
            }
            if (typeof price !== 'number' || price <= 0) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidPrice);
                return response;
            }
            if (!validTimeInForce.includes(timeInForce)) {
                response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidTimeInForce);
                return response;
            }
            let quantityToTrade = size;
            let sideToProcess;
            let sideToAdd;
            let comparator;
            let iter;
            if (side === side_1.Side.BUY) {
                sideToAdd = this.bids;
                sideToProcess = this.asks;
                comparator = this.greaterThanOrEqual;
                iter = this.asks.minPriceQueue;
            }
            else {
                sideToAdd = this.asks;
                sideToProcess = this.bids;
                comparator = this.lowerThanOrEqual;
                iter = this.bids.maxPriceQueue;
            }
            if (timeInForce === order_1.TimeInForce.FOK) {
                const fillable = this.canFillOrder(sideToProcess, side, size, price);
                if (!fillable) {
                    response.err = (0, errors_1.CustomError)(errors_1.ERROR.ErrLimitFOKNotFillable);
                    return response;
                }
            }
            let bestPrice = iter();
            while (quantityToTrade > 0 &&
                sideToProcess.len() > 0 &&
                bestPrice !== undefined &&
                comparator(price, bestPrice.price())) {
                const { done, partial, partialQuantityProcessed, quantityLeft } = this.processQueue(bestPrice, quantityToTrade);
                response.done = response.done.concat(done);
                response.partial = partial;
                response.partialQuantityProcessed = partialQuantityProcessed;
                quantityToTrade = quantityLeft;
                response.quantityLeft = quantityToTrade;
                bestPrice = iter();
            }
            if (quantityToTrade > 0) {
                const order = new order_1.Order(orderID, side, new bignumber_js_1.default(quantityToTrade), price, Date.now(), true);
                if (response.done.length > 0) {
                    response.partialQuantityProcessed = size - quantityToTrade;
                    response.partial = order;
                }
                this.orders[orderID] = sideToAdd.append(order);
            }
            else {
                let totalQuantity = 0;
                let totalPrice = 0;
                response.done.forEach((order) => {
                    const ordrSize = order.size.toNumber();
                    totalQuantity += ordrSize;
                    totalPrice += order.price * ordrSize;
                });
                if (response.partialQuantityProcessed > 0 && response.partial !== null) {
                    totalQuantity += response.partialQuantityProcessed;
                    totalPrice +=
                        response.partial.price * response.partialQuantityProcessed;
                }
                response.done.push(new order_1.Order(orderID, side, new bignumber_js_1.default(size), totalPrice / totalQuantity, Date.now()));
            }
            // If IOC order was not matched completely remove from the order book
            if (timeInForce === order_1.TimeInForce.IOC && response.quantityLeft > 0) {
                this.cancel(orderID);
            }
            return response;
        };
        /**
         * Modify an existing order with given ID
         *
         * @param orderID - The ID of the order to be modified
         * @param orderUpdate - An object with the modified size and/or price of an order. To be note that the `side` can't be modified. The shape of the object is `{side, size, price}`.
         * @returns The modified order if exists or `undefined`
         */
        this.modify = (orderID, orderUpdate) => {
            const order = this.orders[orderID];
            if (order === undefined)
                return;
            const side = orderUpdate.side;
            if (side === side_1.Side.BUY) {
                return this.bids.update(order, orderUpdate);
            }
            else if (side === side_1.Side.SELL) {
                return this.asks.update(order, orderUpdate);
            }
            else {
                throw (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidSide);
            }
        };
        /**
         * Remove an existing order with given ID from the order book
         *
         * @param orderID - The ID of the order to be removed
         * @returns The removed order if exists or `undefined`
         */
        this.cancel = (orderID) => {
            const order = this.orders[orderID];
            if (order === undefined)
                return;
            /* eslint-disable @typescript-eslint/no-dynamic-delete */
            delete this.orders[orderID];
            if (order.side === side_1.Side.BUY) {
                return this.bids.remove(order);
            }
            return this.asks.remove(order);
        };
        /**
         * Get an existing order with the given ID
         *
         * @param orderID - The ID of the order to be returned
         * @returns The order if exists or `undefined`
         */
        this.order = (orderID) => {
            return this.orders[orderID];
        };
        // Returns price levels and volume at price level
        this.depth = () => {
            const asks = [];
            const bids = [];
            this.asks.priceTree().forEach((levelPrice, level) => {
                asks.push([levelPrice, level.volume().toNumber()]);
            });
            this.bids.priceTree().forEach((levelPrice, level) => {
                bids.push([levelPrice, level.volume().toNumber()]);
            });
            return [asks, bids];
        };
        this.toString = () => {
            return (this.asks.toString() +
                '\r\n------------------------------------' +
                this.bids.toString());
        };
        // Returns total market price for requested quantity
        // if err is not null price returns total price of all levels in side
        this.calculateMarketPrice = (side, size) => {
            let price = 0;
            let err = null;
            let level;
            let iter;
            if (side === side_1.Side.BUY) {
                level = this.asks.minPriceQueue();
                iter = this.asks.greaterThan;
            }
            else {
                level = this.bids.maxPriceQueue();
                iter = this.bids.lowerThan;
            }
            while (size > 0 && level !== undefined) {
                const levelVolume = level.volume().toNumber();
                const levelPrice = level.price();
                if (this.greaterThanOrEqual(size, levelVolume)) {
                    price += levelPrice * levelVolume;
                    size -= levelVolume;
                    level = iter(levelPrice);
                }
                else {
                    price += levelPrice * size;
                    size = 0;
                }
            }
            if (size > 0) {
                err = (0, errors_1.CustomError)(errors_1.ERROR.ErrInsufficientQuantity);
            }
            return { price, err };
        };
        this.greaterThanOrEqual = (a, b) => {
            return a >= b;
        };
        this.lowerThanOrEqual = (a, b) => {
            return a <= b;
        };
        this.processQueue = (orderQueue, quantityToTrade) => {
            const response = {
                done: [],
                partial: null,
                partialQuantityProcessed: 0,
                quantityLeft: quantityToTrade,
                err: null
            };
            if (response.quantityLeft > 0) {
                while (orderQueue.len() > 0 && response.quantityLeft > 0) {
                    const headOrder = orderQueue.head();
                    if (headOrder !== undefined) {
                        const headSize = headOrder.size.toNumber();
                        if (response.quantityLeft < headSize) {
                            response.partial = new order_1.Order(headOrder.id, headOrder.side, new bignumber_js_1.default(headSize - response.quantityLeft), headOrder.price, headOrder.time, true);
                            this.orders[headOrder.id] = response.partial;
                            response.partialQuantityProcessed = response.quantityLeft;
                            orderQueue.update(headOrder, response.partial);
                            response.quantityLeft = 0;
                        }
                        else {
                            response.quantityLeft = response.quantityLeft - headSize;
                            const canceledOrder = this.cancel(headOrder.id);
                            if (canceledOrder !== undefined)
                                response.done.push(canceledOrder);
                        }
                    }
                }
            }
            return response;
        };
        this.canFillOrder = (orderSide, side, size, price) => {
            return side === side_1.Side.BUY
                ? this.buyOrderCanBeFilled(orderSide, size, price)
                : this.sellOrderCanBeFilled(orderSide, size, price);
        };
        this.buyOrderCanBeFilled = (orderSide, size, price) => {
            const insufficientSideVolume = orderSide.volume().lt(size);
            if (insufficientSideVolume) {
                return false;
            }
            let cumulativeSize = 0;
            orderSide.priceTree().forEach((_key, priceLevel) => {
                if (price >= priceLevel.price() && cumulativeSize < size) {
                    const volume = priceLevel.volume().toNumber();
                    cumulativeSize += volume;
                }
                else {
                    return true; // break the loop
                }
            });
            return cumulativeSize >= size;
        };
        this.sellOrderCanBeFilled = (orderSide, size, price) => {
            const insufficientSideVolume = orderSide.volume().lt(size);
            if (insufficientSideVolume) {
                return false;
            }
            let cumulativeSize = 0;
            orderSide.priceTree().forEach((_key, priceLevel) => {
                if (price <= priceLevel.price() && cumulativeSize < size) {
                    const volume = priceLevel.volume().toNumber();
                    cumulativeSize += volume;
                }
                else {
                    return true; // break the loop
                }
            });
            return cumulativeSize >= size;
        };
        this.getOrderByOrderId = (orderId) => {
            if (this.orders[`${orderId}`])
                return this.orders[`${orderId}`];
            return null;
        };
        this.bids = new orderside_1.OrderSide(side_1.Side.BUY);
        this.asks = new orderside_1.OrderSide(side_1.Side.SELL);
    }
}
exports.OrderBook = OrderBook;
