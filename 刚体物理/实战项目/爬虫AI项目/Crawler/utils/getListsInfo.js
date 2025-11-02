const puppeteer=require('puppeteer');
const {stop}=require('./stop')
exports.getListsInfo=async function getListsInfo(url){
    const browser=await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(url,{waitUntil: 'domcontentloaded',timeout:180000});
    await stop(5000)
    const result = await page.evaluate(function(){
        let ifameDoc=document.querySelector('iframe#g_iframe').contentDocument;
        let topListsCategory=[];
        let globalTopLists=ifameDoc.querySelectorAll('ul.f-cb')[1];
        globalTopLists=globalTopLists.querySelectorAll('li');
        globalTopLists.forEach((item)=>{
	    topListsCategory.push({
        name:item.querySelectorAll('a')[1].innerText,
        listLink:item.querySelector('a').href})
    })
     return topListsCategory;
    });
    browser.close();
    return result;
}


