const puppeteer=require('puppeteer');
const {stop}=require('./stop')
exports.getSongsInfoByCategory=async function getSongsInfoByCategory(url){
    const browser=await puppeteer.launch();
    try{
        
        const page = await browser.newPage();
        await page.goto(url,{waitUntil: 'domcontentloaded'});
        await stop(5000)
        const result = await page.evaluate(function(){
            let ifameDoc=document.querySelector('iframe#g_iframe').contentDocument
            let topSongs=ifameDoc.querySelectorAll('span.txt')
            let topSongInfo=[];
            topSongs.forEach((song)=>{topSongInfo.push({
                songName:song.querySelector('b').title,
                songLink:song.querySelector('a').href
            })})
            return topSongInfo;
        });
        browser.close();
        return result;
    }catch(err){
        return undefined;
    }finally{
        await browser.close()
    }
}

