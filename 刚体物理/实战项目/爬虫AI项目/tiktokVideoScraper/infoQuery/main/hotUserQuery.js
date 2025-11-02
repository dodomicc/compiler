const querystring = require('querystring');
const tackleAntiScraper=require('../../tacklsAntiScraper/main/tackleAntiScraper')
const getWholePage=require('../utils/getWholePage')
const extractHotUsers=require('../utils/extractHotUsers')
module.exports=async function getHotUsersByQuery(query,page){
    const url=getQueryURL(query);
    await page.goto(url,{timeout:60000});
    await tackleAntiScraper(page);
    await getWholePage(page);
    const hotUsers=extractHotUsers(page);
    return hotUsers;
}



function getQueryURL(query){
    const urlPrefix = 'https://www.tiktok.com/search/user?';
    const queryParams = {
        q: query,
    };
    const queryString = querystring.stringify(queryParams);
    return urlPrefix + queryString;
}






