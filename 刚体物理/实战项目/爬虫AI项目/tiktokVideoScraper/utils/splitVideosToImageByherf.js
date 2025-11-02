const getVideoOriginalURL = require("./getVideoOriginalURL")
const ffmpeg=require('fluent-ffmpeg')
const fs=require('fs')
const axios=require('axios');
const { timeStamp } = require("console");
module.exports=async function splitVideosToImageByherf(href,outputFolder,videoId){
    try{
        const originalURL=await getVideoOriginalURL(href,10);
        const originalVideoLocation=outputFolder+`/${videoId}.mp4`;
        console.log('开始下载视频')
        await downloadVideo(originalURL.videoURL,originalVideoLocation);
        console.log('视频下载成功')
        // ffmpeg(originalVideoLocation)
        // .screenshots({
        //     folder: outputFolder,   // 截图保存的文件夹路径
        //     filename: 'image-%s.png',// 截图文件名格式
        //     timeStamp:[5,10,20]
        // })
        // // 执行截图任务
        // .on('end', function() {
        //     console.log('截图完成');
        // })
        // .on('error', function(err) {
        //     console.error('截图失败: ', err);
        // });

       
  
    }catch(err){
        console.log(err);
    }
}

async function downloadVideo(url,outputLocation){
    return new Promise((resolve,reject)=>{
        const writer=fs.createWriteStream(outputLocation);
        axios({
            url:url,
            method:'get',
            responseType:'stream'
        }).then(res=>{
            res.data.pipe(writer);
            res.data.on('end',resolve)
        }).catch(err=>{
            reject();
        })
       
    })
}