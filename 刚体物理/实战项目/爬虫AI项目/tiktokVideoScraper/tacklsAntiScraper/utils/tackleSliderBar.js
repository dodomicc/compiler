const getSliderStart=require('./getSliderStart')
module.exports=async function tackleSliderBar(page){
    const imgMeta=await page.evaluate(()=>{
        const img=document.querySelector('#captcha-verify-image');
        return {src:img.src,width:img.width,height:img.height,
            sliderBarWidth:document.querySelector('img.captcha_verify_img_slide').width
        }
    })
    const moveLength=await getSliderStart(imgMeta.src,imgMeta.width,imgMeta.height,imgMeta.sliderBarWidth)
    await page.waitForFunction(async (moveLength)=>{
        const delay=async function (waitTime){
            return new Promise(resolve=>{
                setTimeout(resolve,waitTime)
            })
        }
        const target=document.querySelector('div.secsdk-captcha-drag-icon');
        const box=target.getBoundingClientRect();
        const startX = box.x; 
        const startY = box.y + box.height / 2; 
        const endX = startX + moveLength; 
        await target.dispatchEvent(new MouseEvent('mousedown',{bubbles:true}));
        await target.dispatchEvent(new MouseEvent('mousemove', {  clientX: startX, clientY: startY}));
        for (let i = 0; i <= 100; i++) {
            const x =(endX - startX) * (i / 100);
            const y = startY;
            await target.dispatchEvent(new MouseEvent('mousemove', {  clientX: x, clientY: y }));
            await delay(20);
        }
        target.dispatchEvent(new MouseEvent('mouseup'));
        return true;
    },{},moveLength)
}