const fs =require('fs')
const ytdl=require('ytdl-core')

async function downloadVideo(url,description){
    return new Promise(function (resolve,reject){
        const stream=ytdl(url)
        const writer=fs.createWriteStream('./video.mp4')
        stream.pipe(writer);
        stream.on('end', () => {
            console.log('下载完成！');
            return resolve({status:'success'})
        });

        // 错误事件
        stream.on('error', (err) => {
            console.error('下载出错:', err);
            return reject('本图片下载失败')
        });
    })
}
function main(){
    const result=downloadVideo('https://www.youtube.com/watch?v=RYjAAiEEAeM','')
    console.log(result);
    
  
}
main()
