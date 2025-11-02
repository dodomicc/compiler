
const tackleAntiScraper=require('../../tacklsAntiScraper/main/tackleAntiScraper')
const getWholePage=require('../utils/getWholePage')
const extractVideos=require('../utils/extractVideos')
const getFullInforVideos=require('../utils/getFullInforVideos')
module.exports=async function getVideoMetaInfoByUID(userId,page){
    const url=getQueryURL(userId);
    await page.goto(url,{timeout:60000});
    await tackleAntiScraper(page);
    await getWholePage(page);
    const allVideosLinks=await extractVideos(page,userId);
    const result= getFullInforVideos(allVideosLinks);
    return result;
}

function getQueryURL(userId){
    return `https://www.tiktok.com/@${userId}`;
}