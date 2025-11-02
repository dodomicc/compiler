const axios = require('axios')
const cheerio = require('cheerio')
const uniqueFilename = require('unique-filename')
const fs = require('fs')
const path = require('path')
const headers = {
  'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
  Referer: 'https://www.tiktok.com/',
}
module.exports.downloadTikTokVideo = async (videoUrl,downloadFolder) => {
  return new Promise(async (resolve, reject) => {
    try {
      //For mobile links
      if (videoUrl.includes('https://vm.tiktok.com')) {
        const videoReqToHtml = await axios.get(videoUrl, {
          headers: headers,
          responseType: 'json',
        })

        const htmlContent = videoReqToHtml.data
        const $ = cheerio.load(htmlContent)

        // __UNIVERSAL_DATA_FOR_REHYDRATION__ script tag
        const scriptTag = $('script#__UNIVERSAL_DATA_FOR_REHYDRATION__')

        // Get data from script tag
        const jsonData = JSON.parse(scriptTag.html())

        const videoWebUrl =
          jsonData['__DEFAULT_SCOPE__']['seo.abtest'].canonical
        const filePath = await videoDownloader(videoWebUrl,downloadFolder)
        resolve(filePath)
      }
      videoUrl.includes('https://www.tiktok.com/') ? resolve(await videoDownloader(videoUrl,downloadFolder)) : undefined;

    } catch (error) {
      reject(error)
    }
  })
}

const videoUrlShorter = async (url) => {
  const startIndex = 0
  const endIndex = 19
  const videoId = url.split('/')[5].substring(startIndex, endIndex)
  return videoId
}

const videoDownloader = async (videoUrl,downloadFolder) => {
  return new Promise(async (resolve, reject) => {
    try {
      const videoId = await videoUrlShorter(videoUrl)
      const downloadUrl =
        'https://api16-normal-c-useast1a.tiktokv.com/aweme/v1/feed/?aweme_id=' +
        videoId
      const response = await axios.get(downloadUrl, {
        headers: headers,
        responseType: 'json',
      })
      const embedVideoUrl =
        response.data.aweme_list[0].video.play_addr.url_list[0]
      const userPath = path.resolve(downloadFolder)
      if (!fs.existsSync(userPath)) {
        fs.mkdirSync(userPath, { recursive: true })
      }
      //Unique name generator for files
      const filePath = `${uniqueFilename(userPath)}.mp4`

      const videoStream = await axios.get(embedVideoUrl, {
        headers: headers,
        responseType: 'stream',
      })
      // returns something like: 'Downloads/51a7b48d.mp4'
      const writeStream = fs.createWriteStream(filePath)
      await videoStream.data.pipe(writeStream)

      writeStream.on('finish', async () => {
        resolve(filePath)
      })
    } catch (error) {
      reject(error)
      console.log(error)
    }
  })
}
