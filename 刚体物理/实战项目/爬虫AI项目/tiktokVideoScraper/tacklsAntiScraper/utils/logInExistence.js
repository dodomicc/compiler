module.exports=async function logInExistence(page){
    const result= await page.evaluate(()=>{
        return document.querySelector('#loginContainer > div > div > div.css-txolmk-DivGuestModeContainer.exd0a435 > div > div.css-u3m0da-DivBoxContainer.e1cgu1qo0 > div > div > div')!=null?true:false;
    })
    return result;
}