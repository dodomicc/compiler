module.exports=async function tackleLogIn(page){
    await page.evaluate(function(){
        document.querySelector(
            '#loginContainer > div > div > div.css-txolmk-DivGuestModeContainer.exd0a435 > div > div.css-u3m0da-DivBoxContainer.e1cgu1qo0 > div > div > div'
        ).click();
    })
}