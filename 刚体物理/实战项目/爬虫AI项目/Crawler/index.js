const {getSongsInfoByCategory}=require('./utils/getSongsInfoByCategory')
const {getListsInfo}=require('./utils/getListsInfo')
const {writeSongsIntoFile}=require('./utils/writeSongsIntoFile')
const puppeteer=require('puppeteer');
const {stop}=require('./utils/stop')
async function writeALLTopSons(){
    console.log('加载开始')
    console.log('------------------------------------------------------------')
    console.log(`热歌榜元信息开始加载`)
    const ListInfo=await  getListsInfo('https://music.163.com/#/discover/toplist?id=8702982391');
    if(ListInfo){
        console.log(`热歌榜元信息加载完成`)
    }else{
        console.log(`热歌榜元信息加载失败`)
    }
    for(var i=0; i<ListInfo.length; i++){
        console.log('------------------------------------------------------------')
        let count=1;
        console.log(`${ListInfo[i].name}开始第${count}次加载`)
        let songs=await getSongsInfoByCategory(ListInfo[i].listLink)
        while(!songs && count<=5) {
            count++
            songs=await getSongsInfoByCategory(ListInfo[i].listLink)
            console.log(`${ListInfo[i].name}加载失败`)
            console.log('------------------------------------------------------------')
            if(count<=5) console.log(`${ListInfo[i].name}开始第${count}次重新加载`)
        }
        if(!songs) continue;
        writeSongsIntoFile(songs,ListInfo,i);
        console.log(`${ListInfo[i].name}加载成功`)

    }
}
// writeALLTopSons()


async function kk(url){
    const browser=await puppeteer.launch();
    try{
        
        const page = await browser.newPage();

        await page.goto(url,{waitUntil: 'networkidle0',timeout:60000});
       
        await stop(20000);
        const result1 = await page.evaluate(function(){
           window.scrollTo(0,document.querySelector('ytd-app').getBoundingClientRect().height)
           return document.querySelector('ytd-app').getBoundingClientRect().height;
        });
        console.log(result1);
      
        await stop(20000);
       
        const result2 = await page.evaluate(function(){
            // window.scrollTo(0,document.querySelector('ytd-app').getBoundingClientRect().height)
            return document.querySelector('ytd-app').getBoundingClientRect().top;
        });
        console.log(result2);
        browser.close();
        return {result1,result2};
    }catch(err){
        console.log(err);
    }finally{
        await browser.close()
    }
}

async function main(url){
    const result=await kk(url);
    console.log(result);
}

main('https://www.youtube.com/');

















