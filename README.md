# EqWS

These are temporary docs. Full versions will be available once a more stable version is established (said every developer ever). Due to active development, we recommend forking the repository to prevent changes in syntax from confounding users. EqWS was previously called DHC and is currently actively developed at https://github.com/andrew-saydjari/DHC.

## Example

Please find an example notebook that runs DHC on image slices loaded from FITS files in from_cannon/2021_02_05. This is an example of the parallelized code suitable for computation across many cores on a single node (here 30). In this example, we calculated 2542 coeff per image on 800230 images in <2.5h. 

## Workflow

The main workflow (as of 2021/02/10) is to compute the filterbank using DHC.fink_filter_hash and then call DHC_compute(image,filterbank,filterbank).

The DHC currently computes second-order convolutions in direct analogy with WST, but always sums power (abs^2) over images rather than just summing the abs. Other second-order couplings are under development and are experimental. Passing the filterbank twice allows the user to specify a different filterbank for the first and second convolutions. This has not yielded particularly compelling results and will be deprecated (or at least made an implicit argument) in the near future.

## Method Parameters

Most of the parameters users will want to adjust are found in the compilation of the filterbank. A standard call looks like

`
filter_hash = fink_filter_hash(1,8,nx=256,wd=2)
`

The first argument (c) specifies the step size of scale (J) for the wavelets. For c=1, there are J = log_2(N_pixels)-2 radial bins. One bin near the origin of Fourier space is excluded because it is not well sampled (and it is compensated for by a catch all filter phi) and one bin near the edges of the Fourier plane is also excluded because it is not well sampled in real space. For c = 2, there are 2xJ radial bins, a half-dyadic spacing (higher c are implemented for those interested, but generally does seem to significantly improve the statistics at least for small images ~256^ pixels).

The second argument is L, the number of angular bins. Default should be L=8. We do sometimes run with L=16. 

The size of the image is assumed 256, but can be changed by passing nx=N_pix.

By default, our wavelets only have nearest neighbor overlap (wd=1), but we often find wavelets more extended in the angular direction improve performance (wd=2). This will be discussed in a forthcoming publication which should release the code and improve the docs.

Currently, pc=1 by default meaning that the L angular bins only tile 0 to \pi. However, pc=2 allows a full tiling 0 to 2\pi with L bins.

Other experimental features are generally breifly described under the function definition but should be viewed as highly experimental.

## Development

Please feel free to post issues and questions about the package or contact andrew.saydjari@cfa.harvard.edu directly. Your contributions are welcome. Just fork the repository and submit a pull request.

## FAQ

We are actively developing GPU capabilities but they are not yet implemented. Sparse 3D wavelets are also in process.
