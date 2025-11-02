const puppeteer=require('puppeteer');
const getHotUsersByQuery=require('./infoQuery/main/hotUserQuery')
const getVideoMetaInfoByUID=require('./infoQuery/main/getVideoMetaInfoByUID')
const getVideoOriginalURL=require('./utils/getVideoOriginalURL')
const splitVideosToImageByherf=require('./utils/splitVideosToImageByherf')
const delay=require('./utils/delay')
async function main(){
    const browser=await puppeteer.launch({
        headless:false
    });
    try{
        const page=await browser.newPage();
        const result=await getVideoMetaInfoByUID('tj_wonderfullandscape',page)
        console.log(result);
        // const result=await getVideoOriginalURL('https://www.tiktok.com/@wzashow/video/7325456976538799402',10)
        // await page.goto('https://chat.openai.com')
        // await delay(3000000)
        // await splitVideosToImageByherf('https://www.tiktok.com/@tj_wonderfullandscape/video/7282407445580877089','./splitedImgByVideos','d6ee2719337882bfc7a37ece9eb52eaf');
       
       
        browser.close();
    }catch(err){
        console.log(err);
        browser.close();
        return undefined;
    }
}
main();






