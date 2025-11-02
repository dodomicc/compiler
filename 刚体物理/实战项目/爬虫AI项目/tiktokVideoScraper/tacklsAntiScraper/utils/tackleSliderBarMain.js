const sliderBarExistence=require('./sliderBarExistence')
const tackleSliderBar=require('./tackleSliderBar')
module.exports=async function tackleSliderBarMain(page,maxRetryLimit){
    let count=0;
    let sliderExistence=await sliderBarExistence(page)
    while(sliderExistence && count<maxRetryLimit){
        await tackleSliderBar(page);
        count++
        sliderExistence=await sliderBarExistence(page);

        if(!sliderExistence){
            console.log('滑动验证框已破解')
        }else{
            console.log(`滑动验证框第${count}次破解失败`)
        }
    }
    if(sliderExistence) console.log('滑动验证框破解失败')
    return sliderExistence?fallse:true;
}