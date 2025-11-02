export interface CreateInstructionResult {
    done: string[]
    partial: string | undefined
    partialQuantityProcessed: number
    quantityLeft: number
    err: string|undefined
}

export interface ModifyInstructionResult{
    orderID:string,
    isSucess:boolean
}

export interface CancelInstructionResult{
    orderID:string,
    isSucess:boolean
}