import BigNumber from "bignumber.js";
import { OrderBook } from "./OrderBook/orderbook";
import { IOrder, Order} from "./OrderUtils/order";
import { Side } from "./OrderUtils/side";
import { CancelInstruction, CreateInstruction, InstructionType, ModifyInstruction } from "./Type/Instructions";
import { CancelInstructionResult, CreateInstructionResult, ModifyInstructionResult } from "./Type/InstructionResult";


export interface MatchEngineSnapShot{
    data:IOrder[]
}

export interface Instructions {
    instructions: (CreateInstruction|ModifyInstruction|CancelInstruction)[]
}

export interface InstructionsResult {
    instructionsResult:(CreateInstructionResult|ModifyInstructionResult|CancelInstructionResult)[]
}

export class MatchEngineExecutor{
    orderBook:OrderBook
    constructor(){

    }
    resetOrderBook=():void=>{
        this.orderBook=new OrderBook();
    }
    getSnapShot=():MatchEngineSnapShot=>{
        const orders=this.orderBook.getOrders();
        const data:IOrder[]=[];
        Object.keys(orders).forEach((key)=>{
            const order=orders[key];
            data.push({
                id: order.id,
                side: order.side,
                size: order.size.toNumber(),
                price: order.price,
                time: order.time,
                isMaker:true
            })
        });
        const snapShot={
            data:data
        }
        return snapShot;
    }
    recoveryFromSnapShot=(recoveryData:MatchEngineSnapShot):void=>{
        this.orderBook=new OrderBook();
        const bids=this.orderBook.getBids();
        const asks=this.orderBook.getAsks();
        recoveryData.data.map(order=>{
            const recoveryOrder=new Order(
                order.id,
                order.side,
                BigNumber(order.size),
                order.price,
                order.time,
                true
            )
            order.side==Side.BUY?bids.append(recoveryOrder):asks.append(recoveryOrder)
        })
    }

    execTask=(tasks:Instructions):InstructionsResult=>{
        const result=[];
        tasks.instructions.map(instruction=>{
            switch (instruction.instructionType) {
                case InstructionType.CREATE:
                    const createOutput=this.orderBook.createOrder({
                        type: instruction.orderType,
                        side: instruction.side,
                        size: instruction.size,
                        timeInForce: instruction.timeInForce,
                        price: instruction.price,
                        orderID: instruction.id,
                    })
                    result.push({
                        done:createOutput.done.map(order=>{return order.id}),
                        partial: createOutput.partial?.id,
                        partialQuantityProcessed: createOutput.partialQuantityProcessed,
                        quantityLeft: createOutput.quantityLeft,
                        err: createOutput.err?createOutput.err.message:undefined
                    })
                    break;
                case InstructionType.MODIFY:
                    const modifyOutput=this.orderBook.modify(instruction.id,instruction.orderUpdate);
                    result.push(modifyOutput?{
                        orderID:instruction.id,
                        isSucess:true
                    }:{
                        orderID:instruction.id,
                        isSuccess:false
                    })
                    break;
                case InstructionType.CANCEL:
                    const cancelOutput=this.orderBook.cancel(instruction.id);
                    result.push(modifyOutput?{
                        orderID:instruction.id,
                        isSucess:true
                    }:{
                        orderID:instruction.id,
                        isSuccess:false
                    })
                    break;
    
                default:
                    break;
            }
        })
        return {
            instructionsResult:result
        }
    }
}
