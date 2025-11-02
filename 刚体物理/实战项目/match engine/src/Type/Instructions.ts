import { OrderType, OrderUpdate, TimeInForce } from "../OrderUtils/order"
import { Side } from "../OrderUtils/side"

export enum InstructionType{
    CREATE='create',
    CANCEL='cancel',
    MODIFY='modify',
}


export interface CreateInstruction{
    instructionType:InstructionType.CREATE
    id: string
    side: Side
    size: number
    price: number|undefined
    time?: number
    timeInForce:TimeInForce|undefined,
    orderType:OrderType
}

export interface CancelInstruction{
    instructionType:InstructionType.CANCEL
    id: string
}

export interface CreateInstruction{
    instructionType:InstructionType.CREATE
    id: string
    side: Side
    size: number
    price: number
    time?: number
    timeInForce:TimeInForce,
    orderType:OrderType
}

export interface ModifyInstruction{
    instructionType:InstructionType.MODIFY
    id: string
    orderUpdate:OrderUpdate
}