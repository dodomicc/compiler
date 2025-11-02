// 这是一个停时的函数
async function delay(waitTime){
    return new Promise(resolve=>{
        setTimeout(resolve,waitTime)
    })
}
module.exports=delay;