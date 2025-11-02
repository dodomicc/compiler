"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MatchEngine = exports.MatchEngineSnapShot = void 0;
const bignumber_js_1 = __importDefault(require("bignumber.js"));
const orderbook_1 = require("./OrderBook/orderbook");
class MatchEngineSnapShot {
}
exports.MatchEngineSnapShot = MatchEngineSnapShot;
class MatchEngine {
    constructor() {
        this.resetOrderBook = () => {
            this.orderBook = new orderbook_1.OrderBook();
        };
        this.getSnapShot = () => {
            const orders = this.orderBook.getOrders();
            const data = [];
            Object.keys(orders).forEach((key) => {
                const order = orders[key];
                data.push({
                    id: order.id,
                    side: order.side,
                    size: order.size.toNumber(),
                    price: order.price,
                    time: order.time,
                    isMaker: true
                });
            });
            const snapShot = {
                data: data
            };
            return snapShot;
        };
    }
}
exports.MatchEngine = MatchEngine;
console.log((0, bignumber_js_1.default)(1 / 7));
console.log((0, bignumber_js_1.default)((0, bignumber_js_1.default)(1 / 7).toNumber()));
