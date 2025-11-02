import {v4 as uuidv4}from 'uuid'
import ReadLocksContainer from './ReadLocksContainer'

export default class ReadLock{

    rawLock:string
    localLock:string
    modeKey:string
    readLockSetKey:string
    lockReleaseChannel:string
    connectionConfig:any
    subscriber:any
    tryAcquireLockResolve:any
    tryAcquireReject:any
    leaseTime:number=30
    tryAcquireTimer=null

    addLockScripts=`if(redis.call('exists',KEYS[1])==0) then 
                        redis.call('set',KEYS[1],ARGV[1])
                        redis.call('sadd',KEYS[2],ARGV[2])
                        redis.call('expire',KEYS[1],ARGV[3])
                        redis.call('expire',KEYS[2],ARGV[3])
                        return nil
                    elseif(redis.call('get',KEYS[1])==ARGV[1]) then 
                        redis.call('sadd',KEYS[2],ARGV[2])
                        redis.call('expire',KEYS[1],ARGV[3])
                        redis.call('expire',KEYS[2],ARGV[3])
                        return nil
                    else
                        return redis.call('ttl',KEYS[1]);
                    end`

    watchDogScripts=`if(redis.call('SISMEMBER',KEYS[2],ARGV[1])==1) then
                        redis.call('expire',KEYS[1],ARGV[2])
                        redis.call('expire',KEYS[2],ARGV[2])
                        return 1
                    else
                        return 0
                    end`

    unlockScripts=`if(redis.call('SISMEMBER',KEYS[2],ARGV[1])==1) then 
                        redis.call('SREM',KEYS[2],ARGV[1])
                        if(redis.call('SCARD',KEYS[2])==0) then 
                            redis.call('DEL',KEYS[1])
                        end
                        redis.call('PUBLISH',ARGV[2],'release')
                    end`

                    
    constructor(rawLock:string,connectionConfig:any,leaseTime?:number){
        this.rawLock=rawLock;
        this.localLock=`${this.rawLock}:${uuidv4()}:${Date.now()}`;
        this.modeKey=`${this.rawLock}:lockMode`;
        this.readLockSetKey=`${this.rawLock}:readLocks`;
        this.connectionConfig=connectionConfig;
        this.lockReleaseChannel=`${this.rawLock}:release`
        if(leaseTime) this.leaseTime=leaseTime
        if(!global.readLocksContainer) global.readLocksContainer=new ReadLocksContainer(this.connectionConfig,this.leaseTime);   
    }


    lock=async():Promise<any>=>{
        global.readLocksContainer.addReadLocks(this);
        global.readLocksContainer.addSubscription(this.lockReleaseChannel);
        return new Promise((resolve,reject)=>{
            this.tryAcquireLockResolve=resolve;
            this.tryAcquireReject=reject;
            this.tryAcquireReadLock(this);
        })
    }



    unlock=async():Promise<void>=>{
       return new Promise(async (resolve, reject)=>{
        await global.readLocksContainer.normalConnection.eval(
            this.unlockScripts,2,this.modeKey,this.readLockSetKey,this.localLock,this.lockReleaseChannel   
        )
        resolve();
       })
    }



    tryAcquireReadLock=async (self:ReadLock):Promise<any>=>{
        if(!global.readLocksContainer.existsLock(self)) return;
        const tryLockResult=await self.canGetLock(self);
        if(tryLockResult===true){
            if(self.tryAcquireTimer) clearTimeout(self.tryAcquireTimer);
            self.tryAcquireTimer=null
            global.readLocksContainer.deleteReadLocks(self);
            self.watchDog(self);
            self.tryAcquireLockResolve();
        }else{
            global.writeLockContainer.addSubscription(self.lockReleaseChannel);
            if(!self.tryAcquireTimer) self.tryAcquireTimer=setTimeout((self)=>{self.tryAcquireTimer=null;self.tryAcquireReadLock(self)},tryLockResult*1000,self)
        }
        
    }
   


    canGetLock=async(self:ReadLock):Promise<any>=>{
        return new Promise(async (resolve,reject)=>{
            const getLockResult=await global.readLocksContainer.normalConnection.eval(
                self.addLockScripts,2,self.modeKey,self.readLockSetKey,'read',self.localLock,self.leaseTime
            )
            if(getLockResult==null){
                return resolve(true);
            }else{
                return resolve(Math.max(self.leaseTime,Number(getLockResult)));
            }
        })
    }



    watchDog=async(self:ReadLock):Promise<any>=>{
        const renewLockResult=await self.renewReadLockExpireTime(self);
        if(renewLockResult) setTimeout((self)=>{
            self.watchDog(self);
        },self.leaseTime/3,self);
    }



    renewReadLockExpireTime=async(self:ReadLock):Promise<any>=>{
        return new Promise(async (resolve,reject)=>{
            const canRenewLock=await global.readLocksContainer.normalConnection.eval(
                self.watchDogScripts,2,self.modeKey,self.readLockSetKey,self.localLock,self.leaseTime
            )
            if(canRenewLock==1) resolve(true);
            resolve(false);
        })
    }
    
}


