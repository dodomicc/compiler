# tiktod 
Wrapper For TikTod API to use download TikTok Post

## Features
- Download TikTok videos
- Download TikTok slide images
- Stalk User
- Random Porn ( Don't do it too often, it's not good XD )

## Installation 
### NPM Version 
```bash
$ npm install tiktod
```
### GitHub Version 
```bash
$ npm install github:fvzy/tiktod
```

## Usage
### Call teh function with variable 
```javascript
const tik = require("tiktod");
```
### Download TikTok Video 
To download TikTok video / Images, use `Download` function with TikTok video URL parameter. 
```javascript
tik.download(url).then(data => {
    console.log(data)
});
```
### Stalk User by Username 
To stalk TikTok user, use `stalk` function with TikTok username parameter. 
```javascript
tik.stalk(username).then(data => {
  console.log(data)
});
```
### Random Porn
To get random porn, use `porn` function without parameter. 
```javascript
tik.porn().then(data => {
  console.log(data)
});
```
