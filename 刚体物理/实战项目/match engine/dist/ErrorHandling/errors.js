"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CustomError = exports.ERROR = void 0;
var ERROR;
(function (ERROR) {
    ERROR["Default"] = "Something wrong";
    ERROR["ErrInsufficientQuantity"] = "orderbook: insufficient quantity to calculate price";
    ERROR["ErrInvalidOrderType"] = "orderbook: supported order type are 'limit' and 'market'";
    ERROR["ErrInvalidPrice"] = "orderbook: invalid order price";
    ERROR["ErrInvalidPriceLevel"] = "orderbook: invalid order price level";
    ERROR["ErrInvalidQuantity"] = "orderbook: invalid order quantity";
    ERROR["ErrInvalidSide"] = "orderbook: given neither 'bid' nor 'ask'";
    ERROR["ErrInvalidTimeInForce"] = "orderbook: supported time in force are 'GTC', 'IOC' and 'FOK'";
    ERROR["ErrLimitFOKNotFillable"] = "orderbook: limit FOK order not fillable";
    ERROR["ErrOrderExists"] = "orderbook: order already exists";
})(ERROR || (exports.ERROR = ERROR = {}));
const CustomError = (error) => {
    switch (error) {
        case ERROR.ErrInvalidQuantity:
            return new Error(ERROR.ErrInvalidQuantity);
        case ERROR.ErrInsufficientQuantity:
            return new Error(ERROR.ErrInsufficientQuantity);
        case ERROR.ErrInvalidPrice:
            return new Error(ERROR.ErrInvalidPrice);
        case ERROR.ErrInvalidPriceLevel:
            return new Error(ERROR.ErrInvalidPriceLevel);
        case ERROR.ErrOrderExists:
            return new Error(ERROR.ErrOrderExists);
        case ERROR.ErrInvalidSide:
            return new Error(ERROR.ErrInvalidSide);
        case ERROR.ErrInvalidOrderType:
            return new Error(ERROR.ErrInvalidOrderType);
        case ERROR.ErrInvalidTimeInForce:
            return new Error(ERROR.ErrInvalidTimeInForce);
        case ERROR.ErrLimitFOKNotFillable:
            return new Error(ERROR.ErrLimitFOKNotFillable);
        default:
            error = error === undefined || error === '' ? '' : `: ${error}`;
            return new Error(`${ERROR.Default}${error}`);
    }
};
exports.CustomError = CustomError;
