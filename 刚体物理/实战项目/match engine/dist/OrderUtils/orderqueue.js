"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrderQueue = void 0;
const bignumber_js_1 = __importDefault(require("bignumber.js"));
const denque_1 = __importDefault(require("denque"));
class OrderQueue {
    constructor(price) {
        // { orderID: index } index in denque
        this._ordersMap = {};
        // returns the number of orders in queue
        this.len = () => {
            return this._orders.length;
        };
        this.toArray = () => {
            return this._orders.toArray();
        };
        // returns price level of the queue
        this.price = () => {
            return this._price;
        };
        // returns price level of the queue
        this.volume = () => {
            return this._volume;
        };
        // returns top order in queue
        this.head = () => {
            return this._orders.peekFront();
        };
        // returns bottom order in queue
        this.tail = () => {
            return this._orders.peekBack();
        };
        // adds order to tail of the queue
        this.append = (order) => {
            this._volume = this._volume.plus(order.size);
            this._orders.push(order);
            this._ordersMap[order.id] = this._orders.length - 1;
            return order;
        };
        // sets up new order to list value
        this.update = (oldOrder, newOrder) => {
            this._volume = this._volume.minus(oldOrder.size).plus(newOrder.size);
            // Remove old order from head
            this._orders.shift();
            /* eslint-disable @typescript-eslint/no-dynamic-delete */
            delete this._ordersMap[oldOrder.id];
            // Add new order to head
            this._orders.unshift(newOrder);
            this._ordersMap[newOrder.id] = 0;
        };
        // removes order from the queue
        this.remove = (order) => {
            this._volume = this._volume.minus(order.size);
            const deletedOrderIndex = this._ordersMap[order.id];
            this._orders.removeOne(deletedOrderIndex);
            delete this._ordersMap[order.id];
            // Update all orders indexes where index is greater than the deleted one
            for (const orderId in this._ordersMap) {
                if (this._ordersMap[orderId] > deletedOrderIndex) {
                    this._ordersMap[orderId] -= 1;
                }
            }
        };
        this.updateOrderSize = (order, size) => {
            const newSize = new bignumber_js_1.default(size);
            this._volume = this._volume.plus(newSize.minus(order.size)); // update volume
            order.size = newSize;
            order.time = Date.now();
        };
        this._price = price;
        this._volume = new bignumber_js_1.default(0);
        this._orders = new denque_1.default();
    }
}
exports.OrderQueue = OrderQueue;
