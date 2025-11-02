"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrderSide = void 0;
const bignumber_js_1 = __importDefault(require("bignumber.js"));
const functional_red_black_tree_1 = __importDefault(require("functional-red-black-tree"));
const errors_1 = require("../ErrorHandling/errors");
const order_1 = require("./order");
const orderqueue_1 = require("./orderqueue");
const side_1 = require("../Type/side");
class OrderSide {
    constructor(side) {
        this._prices = {};
        this._volume = new bignumber_js_1.default(0);
        this._total = new bignumber_js_1.default(0);
        this._numOrders = 0;
        this._depthSide = 0;
        this._side = side_1.Side.SELL;
        // returns amount of orders
        this.len = () => {
            return this._numOrders;
        };
        // returns depth of market
        this.depth = () => {
            return this._depthSide;
        };
        // returns total amount of quantity in side
        this.volume = () => {
            return this._volume;
        };
        // returns the total (size * price of each price level) in side
        this.total = () => {
            return this._total;
        };
        // returns the price tree in side
        this.priceTree = () => {
            return this._priceTree;
        };
        // appends order to definite price level
        this.append = (order) => {
            const price = order.price;
            const strPrice = price.toString();
            if (this._prices[strPrice] === undefined) {
                const priceQueue = new orderqueue_1.OrderQueue(price);
                this._prices[strPrice] = priceQueue;
                this._priceTree = this._priceTree.insert(price, priceQueue);
                this._depthSide += 1;
            }
            this._numOrders += 1;
            this._volume = this._volume.plus(order.size);
            this._total = this._total.plus(order.size.multipliedBy(order.price));
            return this._prices[strPrice].append(order);
        };
        // removes order from definite price level
        this.remove = (order) => {
            const price = order.price;
            const strPrice = price.toString();
            if (this._prices[strPrice] === undefined)
                throw (0, errors_1.CustomError)(errors_1.ERROR.ErrInvalidPriceLevel);
            this._prices[strPrice].remove(order);
            if (this._prices[strPrice].len() === 0) {
                /* eslint-disable @typescript-eslint/no-dynamic-delete */
                delete this._prices[strPrice];
                this._priceTree = this._priceTree.remove(price);
                this._depthSide -= 1;
            }
            this._numOrders -= 1;
            this._volume = this._volume.minus(order.size);
            this._total = this._total.minus(order.size.multipliedBy(order.price));
            return order;
        };
        this.update = (oldOrder, orderUpdate) => {
            var _a;
            if (orderUpdate.price !== undefined &&
                orderUpdate.price !== oldOrder.price) {
                // Price changed. Remove order and update tree.
                this.remove(oldOrder);
                const newOrder = new order_1.Order(oldOrder.id, oldOrder.side, orderUpdate.size !== undefined ? new bignumber_js_1.default(orderUpdate.size) : oldOrder.size, orderUpdate.price, Date.now(), oldOrder.isMaker);
                this.append(newOrder);
                return newOrder;
            }
            else if (orderUpdate.size !== undefined &&
                orderUpdate.size !== oldOrder.size.toNumber()) {
                // Quantity changed. Price is the same.
                const oldOrderSize = oldOrder.size.toNumber();
                const strPrice = oldOrder.price.toString();
                const newOrderPrize = (_a = orderUpdate.price) !== null && _a !== void 0 ? _a : oldOrder.price;
                this._volume = this._volume.plus(orderUpdate.size - oldOrderSize);
                this._total = this._total.plus(orderUpdate.size * newOrderPrize - oldOrderSize * oldOrder.price);
                this._prices[strPrice].updateOrderSize(oldOrder, orderUpdate.size);
                return oldOrder;
            }
        };
        // returns max level of price
        this.maxPriceQueue = () => {
            if (this._depthSide > 0) {
                const max = this._side === side_1.Side.SELL ? this._priceTree.end : this._priceTree.begin;
                return max.value;
            }
        };
        // returns min level of price
        this.minPriceQueue = () => {
            if (this._depthSide > 0) {
                const min = this._side === side_1.Side.SELL ? this._priceTree.begin : this._priceTree.end;
                return min.value;
            }
        };
        // returns nearest OrderQueue with price less than given
        this.lowerThan = (price) => {
            const node = this._side === side_1.Side.SELL
                ? this._priceTree.lt(price)
                : this._priceTree.gt(price);
            return node.value;
        };
        // returns nearest OrderQueue with price greater than given
        this.greaterThan = (price) => {
            const node = this._side === side_1.Side.SELL
                ? this._priceTree.gt(price)
                : this._priceTree.lt(price);
            return node.value;
        };
        // returns all orders
        this.orders = () => {
            let orders = [];
            for (const price in this._prices) {
                const allOrders = this._prices[price].toArray();
                orders = orders.concat(allOrders);
            }
            return orders;
        };
        this.toString = () => {
            let s = '';
            let level = this.maxPriceQueue();
            while (level !== undefined) {
                const volume = level.volume().toString();
                s += `\n${level.price()} -> ${volume}`;
                level = this.lowerThan(level.price());
            }
            return s;
        };
        const compare = side === side_1.Side.SELL
            ? (a, b) => a - b
            : (a, b) => b - a;
        this._priceTree = (0, functional_red_black_tree_1.default)(compare);
        this._side = side;
    }
}
exports.OrderSide = OrderSide;
