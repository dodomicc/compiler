"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Order = exports.TimeInForce = exports.OrderType = void 0;
var OrderType;
(function (OrderType) {
    OrderType["LIMIT"] = "limit";
    OrderType["MARKET"] = "market";
})(OrderType || (exports.OrderType = OrderType = {}));
var TimeInForce;
(function (TimeInForce) {
    TimeInForce["GTC"] = "GTC";
    TimeInForce["IOC"] = "IOC";
    TimeInForce["FOK"] = "FOK";
})(TimeInForce || (exports.TimeInForce = TimeInForce = {}));
class Order {
    constructor(orderId, side, size, price, time, isMaker) {
        // This method returns a string representation of the order
        this.toString = () => (`${this._id}:
    side: ${this._side}
    size: ${this._size.toString()}
    price: ${this._price}
    time: ${this._time}
    isMaker: ${this._isMaker}`);
        // This method returns a JSON string representation of the order
        this.toJSON = () => JSON.stringify({
            id: this._id,
            side: this._side,
            size: this._size.toNumber(),
            price: this._price,
            time: this._time,
            isMaker: this._isMaker
        });
        // This method returns an object representation of the order
        this.toObject = () => ({
            id: this._id,
            side: this._side,
            size: this._size.toNumber(),
            price: this._price,
            time: this._time,
            isMaker: this._isMaker
        });
        this._id = orderId;
        this._side = side;
        this._price = price;
        this._size = size;
        this._time = time !== null && time !== void 0 ? time : Date.now();
        this._isMaker = isMaker !== null && isMaker !== void 0 ? isMaker : false;
    }
    // Getter for order ID
    get id() {
        return this._id;
    }
    // Getter for order side
    get side() {
        return this._side;
    }
    // Getter for order price
    get price() {
        return this._price;
    }
    // Getter for order size
    get size() {
        return this._size;
    }
    // Setter for order size
    set size(size) {
        this._size = size;
    }
    // Getter for order timestamp
    get time() {
        return this._time;
    }
    // Setter for order timestamp
    set time(time) {
        this._time = time;
    }
    // Getter for order isMaker
    get isMaker() {
        return this._isMaker;
    }
}
exports.Order = Order;
