module.exports=async function sliderBarExistence(page){
    const result= await page.evaluate(()=>{
        return document.querySelector('img.captcha_verify_img_slide')!=null?true:false;
    })
    return result;
}