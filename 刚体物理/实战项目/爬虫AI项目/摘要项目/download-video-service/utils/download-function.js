const fs =require('fs')
const ytdl=require('ytdl-core')
const Spark5=require('spark-md5')
exports.downloadingFunc=async function downloadVideo(url,description){
    fileName=Spark5.hash(description);
    return new Promise(function (resolve,reject){
        const stream=ytdl(url)
        const writer=fs.createWriteStream(`./files/${fileName}.mp4`)
        stream.pipe(writer);
        stream.on('end', () => {
            console.log('下载完成！');
            resolve({status:'success'})
        });

        // 错误事件
        stream.on('error', (err) => {
            console.error('下载出错:', err);
            reject('本图片下载失败')
        });
    })
}