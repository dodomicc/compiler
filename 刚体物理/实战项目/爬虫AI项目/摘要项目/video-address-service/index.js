import axios from 'axios'
import he from 'he'
import striptags from 'striptags'
import puppeteer from 'puppeteer'
import Spark5 from 'spark-md5'
import fs from 'fs'



async function delay(waitTime){
    return new Promise(resolve=>{
        setTimeout(()=>{
            resolve(2);
        },waitTime)
    })
}




const getHTML = async (video_URL) => {
    const {data: html} = await axios.get(video_URL);
    return html
}



const getSubtitle = async (html) => {
    if (!html.includes('captionTracks'))  throw new Error(`Could not find captions for video`);
    const regex = /https:\/\/www\.youtube\.com\/api\/timedtext[^"]+/;
    const [url] = html.match(regex);
    if (!url) throw new Error(`Could not find captions`);
    const obj = JSON.parse(`{"url": "${url}"}`)
    const subtitle_url = obj.url
    const transcriptResponse = await axios.get(subtitle_url);
    const transcript = transcriptResponse.data;
    const lines = transcript
        .replace('<?xml version="1.0" encoding="utf-8" ?><transcript>', '')
        .replace('</transcript>', '')
        .split('</text>')
        .filter(line => line && line.trim())
        .map(line => {
        const htmlText = line.replace(/<text.+>/, '').replace(/&amp;/gi, '&').replace(/<\/?[^>]+(>|$)/g, '');
        const decodedText = he.decode(htmlText);
        const text = striptags(decodedText);
        return text;
    });
    console.log('本视频加载成功')
    return lines;
}

async function getContextByVideoURL(video_URL){
    try{
    const html=await getHTML(video_URL);
    const result=await getSubtitle(html);
    return result;
    }catch{
        console.log('本视频字幕不可见，读取失败')
    }
}
// 本函数还有一个问题没有解决，即滚动瀑布流如何自动滑动加载,此问题已经解决
// 哈哈哈



async function getVideoURLByUserId(userPlayListURL){
    const browser=await puppeteer.launch();
    try{
        const page = await browser.newPage();
        await page.goto(userPlayListURL,{timeout:60000});
        let prevHeight=0;
        let currentHeight=500;
        let result=[];
        while(Math.abs(prevHeight-currentHeight)>5){
            await delay(2000);
            prevHeight= await page.evaluate(function(){
                const height=document.querySelector('ytd-app').getBoundingClientRect().height;
                window.scrollTo(0,height)
                return height;
            });
            await delay(2000);
            currentHeight= await page.evaluate(function(){
                const height=document.querySelector('ytd-app').getBoundingClientRect().height;
                window.scrollTo(0,height)
                return height;
            });
            console.log(currentHeight);
        }
        const thumbnails = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('a#video-title-link')).map(a => {return {title:a.title,href:a.href}});
          });
          thumbnails.forEach(thumbnail=>{
            if(thumbnail.title && thumbnail.href) result.push(thumbnail);
            return 
          })
        browser.close();
        return result
    }catch(err){
        console.log(err);
    }finally{
        await browser.close()
    }
   
}
async function getEachVideoCaptionsForUser(userPlayListURL){
    const videos=await getVideoURLByUserId(userPlayListURL);
    const captionsForEachVideo=[];
    let userId=userPlayListURL;
    userId=userId.replace(/^https:\/\/www.youtube.com\/@/,'')
    userId=userId.replace(/\/videos$/,'');
    console.log(userId);
    if(!fs.existsSync(`./videosText/${userId}`))
    fs.mkdirSync(`./videosText/${userId}`, { recursive: true }, (err) => {
        if (err) {
          console.error(`文件夹:./videosText/${userId}无法创建`, err);
          return;
        }
        console.log(`文件夹:./videosText/${userId} 已成功创建`);
    })
    for(var i=0; i<videos.length; i++){
        const captionsForVideo=await getContextByVideoURL(videos[i].href);
        if(captionsForVideo){
            captionsForEachVideo.push(captionsForVideo);
            const fileName=Spark5.hash(videos[i].title)+'.txt'
            captionsForVideo.forEach((line)=>{
                fs.writeFileSync(`./videosText/${userId}/${fileName}`,line+'\n',{flag:'a'})
            })

        }
    }
    
    return captionsForEachVideo;
}




getEachVideoCaptionsForUser('https://www.youtube.com/@wangzhian/videos')

