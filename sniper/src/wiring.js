import { ethers } from 'ethers';
import {
  LAPTOP, WETH, UNI_V3_FACTORY, UNI_V3_QUOTER, UNI_V3_ROUTER,
  AERO_FACTORY, AERO_ROUTER,
  ERC20_ABI, UNI_FACTORY_ABI, UNI_QUOTER_ABI, UNI_ROUTER_ABI,
  AERO_FACTORY_ABI, AERO_ROUTER_ABI
} from './constants.js';

/** Read contracts bind to the provider; the two routers that spend funds bind to the wallet. */
export function buildContracts(provider, wallet) {
  return {
    laptop: new ethers.Contract(LAPTOP, ERC20_ABI, provider),
    laptopWrite: new ethers.Contract(LAPTOP, ERC20_ABI, wallet),
    weth: new ethers.Contract(WETH, ERC20_ABI, provider),
    uniFactory: new ethers.Contract(UNI_V3_FACTORY, UNI_FACTORY_ABI, provider),
    uniQuoter: new ethers.Contract(UNI_V3_QUOTER, UNI_QUOTER_ABI, provider),
    uniRouter: new ethers.Contract(UNI_V3_ROUTER, UNI_ROUTER_ABI, wallet),
    aeroFactory: new ethers.Contract(AERO_FACTORY, AERO_FACTORY_ABI, provider),
    aeroRouterRead: new ethers.Contract(AERO_ROUTER, AERO_ROUTER_ABI, provider),
    aeroRouter: new ethers.Contract(AERO_ROUTER, AERO_ROUTER_ABI, wallet)
  };
}
