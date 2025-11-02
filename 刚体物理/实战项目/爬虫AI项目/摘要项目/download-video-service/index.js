const fs =require('fs')
const downLoad=require('./utils/download-function')
async function main(){
    fs.mkdirSync('./files')
    const result=await downLoad.downloadingFunc('https://www.youtube.com/watch?v=kaLkj-6GDb4&list=PLbiZ073RgtF035gZ-NMQu8SppTmrTk-x9','hello world')
    console.log(result);
}
main()