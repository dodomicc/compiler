const puppeteer=require('puppeteer')

async function delay(waitTime){
    return new Promise(resolve=>{
        setTimeout(()=>{
            resolve(2);
        },waitTime)
    })
}

async function getAllVideosByUserID(url){
    const browser=await puppeteer.launch();
    try{
        const page = await browser.newPage();
        await page.goto(url,{timeout:60000});
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

async function main(url){
    const result=await getAllVideosByUserID(url);
    console.log(result);
}

main('https://www.youtube.com/@wenzhaoofficial/videos');