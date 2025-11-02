const tackleSliderBarMain=require('../utils/tackleSliderBarMain')
const tackeLogInMain=require('../utils/tackeLogInMain')
const delay=require('../../utils/delay')
module.exports=async function tackelAntiScraper(page){
    await delay(3000);
    console.log('--------------------------------------------------------------------------------')
    console.log('开始破解')
    await tackleSliderBarMain(page,10);
    await tackeLogInMain(page,10);
    await delay(3000);
}