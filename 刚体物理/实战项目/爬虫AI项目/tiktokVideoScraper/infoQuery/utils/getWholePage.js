const delay=require('../../utils/delay')
module.exports=async function getWholePage(page){
    let prevHeight=0,curHeight=500;
    await delay(3000);
    console.log('--------------------------------------------------------------------------------')
    console.log('页面开始加载')
    while(prevHeight<curHeight){
        await delay(1000)
        prevHeight=await page.evaluate(function(){
            const height=document.body.getBoundingClientRect().height;
            window.scrollTo(0,height)
            return height
        })
        await delay(1000)
        curHeight=await page.evaluate(function(){
            const height=document.body.getBoundingClientRect().height;
            window.scrollTo(0,height)
            return height
        })
        console.log(`页面高度已更新,当前页面高度为${Math.floor(curHeight)}`)
    }
    console.log('页面加载完成');
    await delay(3000);
}